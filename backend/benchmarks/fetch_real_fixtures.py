"""Download public UCI fixtures; optional retail CSV export requires openpyxl."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/private/tmp/analytico-real-fixtures'))
    parser.add_argument('--datasets', nargs='+', choices=['bike', 'bank', 'retail', 'wine'], default=['bike', 'bank'])
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'evals/real_dataset_manifest.json').read_text())
    specs = {item['id']: item for item in manifest['datasets']}
    for name in args.datasets:
        spec = specs[name]
        archive = args.root / (name + '.zip')
        if not archive.exists():
            with urllib.request.urlopen(spec['download'], timeout=45) as response:
                archive.write_bytes(response.read())
        output = args.root / spec['filename']
        with zipfile.ZipFile(archive) as z:
            if name == 'retail':
                import openpyxl
                book = openpyxl.load_workbook(io.BytesIO(z.read(spec['member'])), read_only=True, data_only=True)
                try:
                    with output.open('w', newline='', encoding='utf-8') as f:
                        csv.writer(f).writerows(book.active.values)
                finally:
                    book.close()
            elif ':' in spec['member']:
                nested_name, member = spec['member'].split(':', 1)
                with zipfile.ZipFile(io.BytesIO(z.read(nested_name))) as nested:
                    output.write_bytes(nested.read(member))
            else:
                output.write_bytes(z.read(spec['member']))
        print(json.dumps({'dataset': name, 'bytes': output.stat().st_size,
                          'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}))

if __name__ == '__main__':
    main()
