#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS Visibility, Centrality & RPKI Validity per ASN from a single RIB (RIPE RIS / RouteViews)

Requires:
  pip install pybgpkit-parser ujson

Example:
  python as_visibility_rpki.py \
    --url https://data.ris.ripe.net/rrc00/2025.10/bview.20251001.0000.gz \
    --vrps /home/jhart/asn-classifier/preprocessing/data/rpki/vrps.json \
    --cache-dir ./bgp_cache \
    --out as_metrics_with_rpki.csv \
    --dedup-per-peer --drop-ends \
    --nonvalidated-dir ./asn_nonvalidated

What you get per ASN (CSV columns):
- seen_peers_path / visibility_path
- seen_peers_origin / visibility_origin
- centrality_mean / endprox_mean / nonterminal_rate
- path_occurrences / origin_occurrences / total_unique_paths / total_peers
- rpki_valid / rpki_invalid_as / rpki_invalid_length / rpki_not_found / rpki_total / rpki_valid_share

Optional side outputs (if --nonvalidated-dir given):
- asn_<ASN>_nonvalidated.txt  (one prefix per line; invalid + not-found)
- asn_<ASN>_invalid.txt
- asn_<ASN>_notfound.txt
"""
import argparse
import csv
import os
import re
import ipaddress
import ujson as json
from collections import defaultdict
from typing import List, Set, Tuple
from pybgpkit_parser import Parser

AS_TOKEN = re.compile(r"[{},()]")  # remove AS_SETs and confed tokens

def parse_as_path(as_path_str: str) -> List[int]:
    if not as_path_str:
        return []
    clean = AS_TOKEN.sub(" ", as_path_str)
    toks = [t for t in clean.split() if t.isdigit()]
    return [int(t) for t in toks]

def norm_pos(i: int, L: int) -> float:
    if L <= 1:
        return 0.0
    return i / (L - 1)

def centrality_from_x(x: float) -> float:
    return max(0.0, 1.0 - 2.0 * abs(x - 0.5))

def end_proximity_from_x(x: float) -> float:
    return min(x, 1.0 - x)

# ---------- RPKI helpers (no pytricia, pure Python) ----------
def load_vrps_simple(path):
    """Load Routinator/rpki-client JSON and build simple dicts:
       { '203.0.113.0/24': [(asn,maxlen), ...], ... } for v4 and v6.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    roas = data.get("roas", data.get("roa", []))
    v4, v6 = {}, {}
    for r in roas:
        asn = int(str(r["asn"]).lstrip("AS"))
        prefix = r["prefix"]
        net = ipaddress.ip_network(prefix)
        maxlen = int(r.get("maxLength", net.prefixlen))
        d = v4 if net.version == 4 else v6
        d.setdefault(net.with_prefixlen, []).append((asn, maxlen))
    return v4, v6

def parents(net):
    """Yield net, net.supernet(/-1), … until /0."""
    cur = net
    yield cur
    while cur.prefixlen > 0:
        cur = cur.supernet(prefixlen_diff=1)
        yield cur

def rpki_validate(prefix, origin_asn, vrp_v4_map, vrp_v6_map):
    """RFC6811-ish:
       - valid: a VRP covers prefix (len <= maxLen) AND ASN matches
       - invalid_as: at least one VRP covers prefix but ASN differs
       - invalid_length: a VRP exists for same ASN but maxLength too small
       - not_found: no covering VRP
    """
    net = ipaddress.ip_network(prefix, strict=False)
    d = vrp_v4_map if net.version == 4 else vrp_v6_map

    candidates = []
    bad_len_same_as = False
    for p in parents(net):
        key = p.with_prefixlen
        if key in d:
            for asn, maxlen in d[key]:
                if net.prefixlen <= maxlen:
                    candidates.append((asn, maxlen))
                if asn == origin_asn and net.prefixlen > maxlen:
                    bad_len_same_as = True

    if candidates:
        return "valid" if any(asn == origin_asn for asn, _ in candidates) else "invalid_as"
    return "invalid_length" if bad_len_same_as else "not_found"

def main():
    ap = argparse.ArgumentParser(description="AS visibility, centrality & RPKI validity per ASN")
    ap.add_argument("--url", required=True, help="URL to a RIB dump (RIPE RIS bview.* or RouteViews rib.*)")
    ap.add_argument("--vrps", required=True, help="Path to vrps.json (Routinator/rpki-client)")
    ap.add_argument("--cache-dir", default="./bgp_cache", help="Local cache dir for pybgpkit (default: ./bgp_cache)")
    ap.add_argument("--out", default="as_metrics_with_rpki.csv", help="Output CSV filename")
    ap.add_argument("--dedup-per-peer", action="store_true", help="Deduplicate identical AS_PATHs per peer")
    ap.add_argument("--min-path-len", type=int, default=3, help="Skip paths shorter than this (default: 3)")
    ap.add_argument("--drop-ends", action="store_true", help="Ignore both ends (neighbor + origin) for centrality")
    ap.add_argument("--exclude-asns", default="", help="Comma-separated ASNs to exclude (e.g., route servers)")
    ap.add_argument("--nonvalidated-dir", default="", help="If set, write per-ASN lists of invalid/not-found prefixes here")
    args = ap.parse_args()

    exclude_asns: Set[int] = set()
    if args.exclude_asns.strip():
        for tok in args.exclude_asns.split(","):
            tok = tok.strip()
            if tok.isdigit():
                exclude_asns.add(int(tok))

    # Prepare optional output dir for non-validated lists
    if args.nonvalidated_dir:
        os.makedirs(args.nonvalidated_dir, exist_ok=True)

    parser = Parser(url=args.url, cache_dir=args.cache_dir)

    # Visibility/Centrality aggregations
    all_peers: Set[str] = set()
    seen_by_path: defaultdict[int, Set[str]] = defaultdict(set)
    seen_by_origin: defaultdict[int, Set[str]] = defaultdict(set)
    cent_sum: defaultdict[int, float] = defaultdict(float)
    endprox_sum: defaultdict[int, float] = defaultdict(float)
    occ_count: defaultdict[int, int] = defaultdict(int)
    nonterminal_hits: defaultdict[int, int] = defaultdict(int)
    seen_paths: Set[Tuple[str, str]] = set()
    total_unique_paths = 0

    # For RPKI: set of unique (prefix, origin_asn)
    route_pairs: Set[Tuple[str, int]] = set()

    for elem in parser:
        peer_id = f"{elem.peer_ip}#{elem.peer_asn}"
        all_peers.add(peer_id)

        # origin_asns direct, fallback via AS_PATH
        origins = []
        if elem.origin_asns:
            for o in elem.origin_asns:
                try:
                    oi = int(o)
                    if oi not in exclude_asns:
                        origins.append(oi)
                except Exception:
                    pass
        if not origins:
            path_fallback = parse_as_path(elem.as_path or "")
            if path_fallback:
                o = path_fallback[-1]
                if o not in exclude_asns:
                    origins = [o]

        # collect origin visibility
        for oasn in origins:
            seen_by_origin[oasn].add(peer_id)

        # collect (prefix, origin) for RPKI later
        if elem.prefix and origins:
            for oasn in origins:
                route_pairs.add((elem.prefix, oasn))

        # path-based metrics (centrality etc.)
        path_list = parse_as_path(elem.as_path or "")
        if not path_list or len(path_list) < args.min_path_len:
            continue

        path_key = " ".join(map(str, path_list))
        dedup_peer = peer_id if args.dedup_per_peer else "*"
        key = (dedup_peer, path_key)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        total_unique_paths += 1

        for asn in set(path_list):
            if asn in exclude_asns:
                continue
            seen_by_path[asn].add(peer_id)

        L = len(path_list)
        start_idx = 1 if args.drop_ends else 0
        end_idx = L - 2 if args.drop_ends else L - 1
        end_positions = {0, L - 1}

        for i, asn in enumerate(path_list):
            if asn in exclude_asns:
                continue
            if i < start_idx or i > end_idx:
                continue

            x = norm_pos(i, L)
            cent_sum[asn] += centrality_from_x(x)
            endprox_sum[asn] += end_proximity_from_x(x)
            occ_count[asn] += 1
            if i not in end_positions:
                nonterminal_hits[asn] += 1

    total_peers = len(all_peers)

    # RPKI: load VRPs and validate all unique (prefix, origin)
    v4map, v6map = load_vrps_simple(args.vrps)
    rpki_stats = defaultdict(lambda: {"valid":0, "invalid_as":0, "invalid_length":0, "not_found":0, "total":0})

    # Optional: collect non-validated per ASN for side files
    nonval_by_asn = defaultdict(list)        # all non-validated prefixes
    invalid_by_asn = defaultdict(list)       # invalid only
    notfound_by_asn = defaultdict(list)      # not-found only

    for prefix, asn in route_pairs:
        res = rpki_validate(prefix, asn, v4map, v6map)
        s = rpki_stats[asn]
        s[res] += 1
        s["total"] += 1
        if args.nonvalidated_dir and res != "valid":
            nonval_by_asn[asn].append(prefix)
            if res == "invalid_as" or res == "invalid_length":
                invalid_by_asn[asn].append(prefix)
            if res == "not_found":
                notfound_by_asn[asn].append(prefix)

    # Write per-ASN non-validated lists if requested
    if args.nonvalidated_dir:
        for asn, lst in nonval_by_asn.items():
            with open(os.path.join(args.nonvalidated_dir, f"asn_{asn}_nonvalidated.txt"), "w") as f:
                for p in sorted(set(lst)): f.write(f"{p}\n")
        for asn, lst in invalid_by_asn.items():
            with open(os.path.join(args.nonvalidated_dir, f"asn_{asn}_invalid.txt"), "w") as f:
                for p in sorted(set(lst)): f.write(f"{p}\n")
        for asn, lst in notfound_by_asn.items():
            with open(os.path.join(args.nonvalidated_dir, f"asn_{asn}_notfound.txt"), "w") as f:
                for p in sorted(set(lst)): f.write(f"{p}\n")

    # Compose CSV
    fieldnames = [
        "asn",
        "seen_peers_path", "visibility_path",
        "seen_peers_origin", "visibility_origin",
        "centrality_mean", "endprox_mean", "nonterminal_rate",
        "path_occurrences", "origin_occurrences",
        "total_unique_paths", "total_peers",
        "rpki_valid", "rpki_invalid_as", "rpki_invalid_length", "rpki_not_found", "rpki_total", "rpki_valid_share",
    ]

    asns = set(seen_by_path) | set(seen_by_origin) | set(occ_count) | set(rpki_stats)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for asn in sorted(asns):
            p_seen = len(seen_by_path.get(asn, set()))
            o_seen = len(seen_by_origin.get(asn, set()))
            occs = occ_count.get(asn, 0)
            cent_mean = (cent_sum.get(asn, 0.0) / occs) if occs else 0.0
            endprox_mean = (endprox_sum.get(asn, 0.0) / occs) if occs else 0.0
            nonterm_rate = (nonterminal_hits.get(asn, 0) / occs) if occs else 0.0

            r = rpki_stats.get(asn, {})
            rv = r.get("valid", 0)
            ria = r.get("invalid_as", 0)
            ril = r.get("invalid_length", 0)
            rnf = r.get("not_found", 0)
            rt = r.get("total", 0)
            rshare = (rv / rt) if rt else 0.0

            w.writerow({
                "asn": asn,
                "seen_peers_path": p_seen,
                "visibility_path": round(p_seen / total_peers, 6) if total_peers else 0.0,
                "seen_peers_origin": o_seen,
                "visibility_origin": round(o_seen / total_peers, 6) if total_peers else 0.0,
                "centrality_mean": f"{cent_mean:.6f}",
                "endprox_mean": f"{endprox_mean:.6f}",
                "nonterminal_rate": f"{nonterm_rate:.6f}",
                "path_occurrences": occs,
                "origin_occurrences": o_seen,
                "total_unique_paths": total_unique_paths,
                "total_peers": total_peers,
                "rpki_valid": rv,
                "rpki_invalid_as": ria,
                "rpki_invalid_length": ril,
                "rpki_not_found": rnf,
                "rpki_total": rt,
                "rpki_valid_share": f"{rshare:.6f}",
            })

    print(f"Wrote {args.out} | ASNs: {len(asns)} | unique_paths: {total_unique_paths} | peers: {total_peers} | pairs: {len(route_pairs)}")

if __name__ == "__main__":
    main()
