import json
from pathlib import Path
import pandas as pd

# locate a likely PeeringDB dump JSON in the workspace
candidates = list(Path('.').glob('*peeringdb*dump*.json')) + list(Path('.').glob('peeringdb*.json'))
if not candidates:
    raise FileNotFoundError("PeeringDB dump JSON not found. Place a file like 'peeringdb_dump.json' in the working directory.")
filepath = candidates[0]

with filepath.open('r', encoding='utf-8') as f:
    dump = json.load(f)

# extract the net.data section and load into a DataFrame
net_data = dump.get('net', {}).get('data')
if net_data is None:
    raise KeyError("JSON does not contain 'net' -> 'data' structure")

net_df = pd.DataFrame(net_data)

# show a quick preview
net_df.head()

import whoisit
from tqdm import tqdm
whoisit.bootstrap()
tqdm.pandas()

net_df['rir'] = net_df['asn'].apply(lambda asn: whoisit.asn(asn).get('rir', 'Unknown'))