import sys
import json
import csv

if __name__ == "__main__":
    # Prüfe, ob Dateipfade als Argumente übergeben wurden
    if len(sys.argv) < 2:
        print("Verwendung: python extract_info_type.py <datei>")
        sys.exit(1)

    file = sys.argv[1]
    with open(file, 'r', encoding='utf-8') as f:
        with open('peeringdb_info_type.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['asn', 'info_type']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            data = json.load(f)
            for item in data['net']['data']:
                if item['info_type'] == '':
                    continue
                writer.writerow({'asn': item.get('asn'), 'info_type': item.get('info_type')})
