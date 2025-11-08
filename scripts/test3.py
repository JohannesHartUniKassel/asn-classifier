import ujson as json
import ipaddress
from collections import defaultdict
from mrtparse import Reader

VRPS_JSON = "/home/jhart/asn-classifier/preprocessing/data/rpki/vrps.json"
MRT_RIB   = "/home/jhart/asn-classifier/scripts/bgp_cache/cache-bview.20251001.0000.a2ce0e1d"

def load_vrps_simple(path):
    """
    Lädt Routinator-/rpki-client-JSON und baut zwei Dicts:
    { "203.0.113.0/24": [(asn,maxlen), ...], ... } jeweils getrennt für v4/v6.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    roas = data.get("roas", data.get("roa", []))  # beide Formate tolerieren
    v4, v6 = {}, {}
    for r in roas:
        asn = int(str(r["asn"]).lstrip("AS"))
        prefix = r["prefix"]
        maxlen = int(r.get("maxLength", ipaddress.ip_network(prefix).prefixlen))
        net = ipaddress.ip_network(prefix)
        d = v4 if net.version == 4 else v6
        d.setdefault(net.with_prefixlen, []).append((asn, maxlen))
    return v4, v6

def normalize_origin(as_path_attr):
    if not as_path_attr:
        return None
    flat = []
    for seg in as_path_attr:
        if seg['type'] in ('AS_SEQUENCE','AS_SET','AS_CONFED_SEQUENCE','AS_CONFED_SET'):
            flat.extend(seg['value'])
    return flat[-1] if flat else None

def mrt_unique_origin_prefixes(mrt_path):
    seen = set()
    r = Reader(mrt_path)
    for m in r:
        if m.err:
            continue
        try:
            bgp = m.mrt.bgp
            if hasattr(bgp, 'rib'):
                prefix = f"{bgp.rib.prefix}/{bgp.rib.plen}"
                for e in bgp.rib.entries:
                    attrmap = {a.type: a for a in e.attr}
                    as_path = None
                    if 17 in attrmap and hasattr(attrmap[17], 'as4_path'):
                        as_path = attrmap[17].as4_path
                    elif 2 in attrmap and hasattr(attrmap[2], 'as_path'):
                        as_path = attrmap[2].as_path
                    origin = normalize_origin(as_path)
                    if origin is None: 
                        continue
                    seen.add((prefix, int(origin)))
        except Exception:
            continue
    return seen

def parents(net):
    """Yield net, net.supernet(/-1), … bis /0."""
    cur = net
    yield cur
    while cur.prefixlen > 0:
        cur = cur.supernet(prefixlen_diff=1)
        yield cur

def rpki_validate(prefix, origin_asn, vrp_v4_map, vrp_v6_map):
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
    v4map, v6map = load_vrps_simple(VRPS_JSON)
    pairs = mrt_unique_origin_prefixes(MRT_RIB)
    stats = defaultdict(lambda: {"valid":0,"invalid_as":0,"invalid_length":0,"not_found":0,"total":0})
    for prefix, asn in pairs:
        res = rpki_validate(prefix, asn, v4map, v6map)
        s = stats[asn]; s[res]+=1; s["total"]+=1
    with open("asn_rpki_stats.csv","w") as f:
        f.write("asn,valid,invalid_as,invalid_length,not_found,total,valid_share\n")
        for asn, s in sorted(stats.items()):
            share = s["valid"]/s["total"] if s["total"] else 0
            f.write(f"{asn},{s['valid']},{s['invalid_as']},{s['invalid_length']},{s['not_found']},{s['total']},{share:.6f}\n")

if __name__ == "__main__":
    main()
