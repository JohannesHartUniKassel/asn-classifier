#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AS Visibility & Centrality from a single RIB dump (RIPE RIS / RouteViews)
Requires: pip install pybgpkit-parser

What you get per ASN (CSV):
- seen_peers_path / visibility_path: share of peers that see the ASN somewhere in AS_PATH
- seen_peers_origin / visibility_origin: share of peers that see the ASN as ORIGIN
- centrality_mean: symmetric "distance-to-middle" score, 0..1 (1 = perfectly central, 0 = at an end)
- endprox_mean: mean(min(x,1-x)), also 0..0.5 (0.5 = central, 0 = at an end)
- nonterminal_rate: fraction of paths where ASN appears NOT at either end (0..1)
- path_occurrences: number of unique paths (after dedup) in which the ASN appears
- origin_occurrences: number of unique paths in which the ASN is origin
- total_unique_paths: total number of unique paths considered (denominator for visibility-based metrics)

Usage example:
  python as_visibility_centrality.py \
    --url https://data.ris.ripe.net/rrc00/2025.10/bview.20251001.0000.gz \
    --out as_metrics.csv --dedup-per-peer --drop-ends

Tips:
- Use --dedup-per-peer to deduplicate identical AS_PATHs per peer (reduces bias).
- Use --drop-ends to ignore both ends (neighbor + origin) when computing centrality/nonterminal_rate.
- Use --exclude-asns 65534,65535 to ignore known route-server/IXP ASNs.
"""
import argparse
import csv
import re
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

def main():
    ap = argparse.ArgumentParser(description="AS Path Visibility + Symmetric Centrality from a single RIB dump")
    ap.add_argument("--url", required=True, help="URL to a RIB dump (e.g., RIPE RIS bview.* or RouteViews rib.*)")
    ap.add_argument("--cache-dir", default="./bgp_cache", help="Local cache dir for downloads (default: ./bgp_cache)")
    ap.add_argument("--out", default="as_metrics.csv", help="Output CSV filename")
    ap.add_argument("--dedup-per-peer", action="store_true",
                    help="Deduplicate identical AS_PATHs per peer to reduce bias (recommended)")
    ap.add_argument("--min-path-len", type=int, default=3, help="Skip paths shorter than this (default: 3)")
    ap.add_argument("--drop-ends", action="store_true",
                    help="Ignore both ends (neighbor + origin) for centrality & nonterminal rate computation")
    ap.add_argument("--exclude-asns", default="", help="Comma-separated ASNs to exclude from ALL metrics (e.g., route servers)")
    args = ap.parse_args()

    exclude_asns: Set[int] = set()
    if args.exclude_asns.strip():
        for tok in args.exclude_asns.split(","):
            tok = tok.strip()
            if tok.isdigit():
                exclude_asns.add(int(tok))

    parser = Parser(url=args.url, cache_dir=args.cache_dir)

    all_peers: Set[str] = set()
    seen_by_path: defaultdict[int, Set[str]] = defaultdict(set)
    seen_by_origin: defaultdict[int, Set[str]] = defaultdict(set)

    cent_sum: defaultdict[int, float] = defaultdict(float)
    endprox_sum: defaultdict[int, float] = defaultdict(float)
    occ_count: defaultdict[int, int] = defaultdict(int)
    nonterminal_hits: defaultdict[int, int] = defaultdict(int)

    seen_paths: Set[Tuple[str, str]] = set()
    total_unique_paths = 0

    for elem in parser:
        peer_id = f"{elem.peer_ip}#{elem.peer_asn}"
        all_peers.add(peer_id)

        for oasn in (elem.origin_asns or []):
            try:
                oasn_int = int(oasn)
            except Exception:
                continue
            if oasn_int in exclude_asns:
                continue
            seen_by_origin[oasn_int].add(peer_id)

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
            cent = centrality_from_x(x)
            endp = end_proximity_from_x(x)

            cent_sum[asn] += cent
            endprox_sum[asn] += endp
            occ_count[asn] += 1

            if i not in end_positions:
                nonterminal_hits[asn] += 1

    total_peers = len(all_peers)

    with open(args.out, "w", newline="") as f:
        fieldnames = [
            "asn",
            "seen_peers_path",
            "visibility_path",
            "seen_peers_origin",
            "visibility_origin",
            "centrality_mean",
            "endprox_mean",
            "nonterminal_rate",
            "path_occurrences",
            "origin_occurrences",
            "total_unique_paths",
            "total_peers",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        asns = set(seen_by_path) | set(seen_by_origin) | set(occ_count)
        for asn in sorted(asns):
            p_seen = len(seen_by_path.get(asn, set()))
            o_seen = len(seen_by_origin.get(asn, set()))
            occs = occ_count.get(asn, 0)
            cent_mean = (cent_sum.get(asn, 0.0) / occs) if occs else 0.0
            endprox_mean = (endprox_sum.get(asn, 0.0) / occs) if occs else 0.0
            nonterm_rate = (nonterminal_hits.get(asn, 0) / occs) if occs else 0.0

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
            })

    print(f"Wrote {args.out} | ASNs: {len(asns)} | unique_paths: {total_unique_paths} | peers: {total_peers}")

if __name__ == "__main__":
    main()