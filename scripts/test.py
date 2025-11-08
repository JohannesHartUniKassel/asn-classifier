# AS Path Visibility pro ASN aus einem einzelnen RIB-Dump (RIPE RIS/RouteViews)
# pip install pybgpkit-parser

from pybgpkit_parser import Parser
import re
import csv

# --- 1) Eingabe: wähle einen RIB-Dump (bview.* von RIS/RouteViews)
# Beispiele:
# RIPE RIS rrc00 (täglich):
# url = "https://data.ris.ripe.net/rrc00/2025.10/bview.20251001.0000.gz"
# RouteViews ORegon-2 (RIB 00:00):
# url = "http://archive.routeviews.org/bgpdata/2025.10/RIBS/rib.20251001.0000.bz2"
# Kleine Demo-Datei (BGPKIT sample):
# url = "https://spaces.bgpkit.org/parser/update-example"
url = "https://data.ris.ripe.net/rrc00/2025.10/bview.20251001.0000.gz"

# --- 2) Parser anlegen (cache_dir spart erneutes Downloaden)
parser = Parser(url=url, cache_dir="./bgp_cache")

# --- 3) Helfer: AS_PATH normalisieren (AS_SETs, Klammern, leere Tokens)
AS_TOKEN = re.compile(r"[{},()]")  # entfernt {1234},(confed),Kommas
def parse_as_path(as_path_str: str):
    if not as_path_str:
        return []
    clean = AS_TOKEN.sub(" ", as_path_str)
    toks = [t for t in clean.split() if t.isdigit()]
    return [int(t) for t in toks]

# --- 4) Aggregation
all_peers = set()               # “peer_id” = "<peer_ip>#<peer_asn>"
seen_by_path = dict()           # asn -> set(peer_id)
seen_by_origin = dict()         # asn -> set(peer_id)

for elem in parser:
    # Peer-ID (kombiniert, damit IPv4/IPv6 eindeutig sind)
    peer_id = f"{elem.peer_ip}#{elem.peer_asn}"
    all_peers.add(peer_id)

    # Origin-ASNs (Liste) – kommt direkt aus dem Parser
    for oasn in elem.origin_asns or []:
        s = seen_by_origin.setdefault(oasn, set())
        s.add(peer_id)

    # Gesamter Pfad
    for asn in parse_as_path(elem.as_path or ""):
        s = seen_by_path.setdefault(asn, set())
        s.add(peer_id)

total_peers = len(all_peers)

# --- 5) Sichtbarkeitsmetrik berechnen und speichern
rows = []
asns = set(seen_by_path) | set(seen_by_origin)
for asn in sorted(asns):
    p_seen = len(seen_by_path.get(asn, set()))
    o_seen = len(seen_by_origin.get(asn, set()))
    rows.append({
        "asn": asn,
        "seen_peers_path": p_seen,
        "visibility_path": round(p_seen / total_peers, 6) if total_peers else 0.0,
        "seen_peers_origin": o_seen,
        "visibility_origin": round(o_seen / total_peers, 6) if total_peers else 0.0,
        "total_peers": total_peers,
    })

out_csv = "as_visibility.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Fertig. {len(rows)} ASNs, total_peers={total_peers}. CSV: {out_csv}")
