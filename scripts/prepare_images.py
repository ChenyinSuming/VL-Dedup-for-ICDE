"""Place the source images of VL-DedupBench under one folder, named by record ID.

The repository does not include images. COCO images come from the COCO 2017 release and
SOP images from the Stanford Online Products release. This script links (or copies) each
image to <out>/<record_id>.jpg so that the pipeline scripts can read it.

Usage:
    python scripts/prepare_images.py --coco-dir COCO2017 --sop-dir Stanford_Online_Products --out resources/images
    python scripts/prepare_images.py --download-coco --sop-dir Stanford_Online_Products --out resources/images

COCO2017 must hold train2017/ and val2017/. With --download-coco, the script fetches the
COCO images that the benchmark uses from images.cocodataset.org instead. The SOP folder is the
unpacked Stanford_Online_Products.zip (it holds bicycle_final/, cabinet_final/, ...).
Add --multiview to also place the extra SOP photos of the multi-view surface.
"""
import argparse
import csv
import gzip
import os
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COCO_URL = 'http://images.cocodataset.org/{}'


def read(path):
    with gzip.open(path, 'rt', newline='') as f:
        return list(csv.DictReader(f))


def place(src, dst, copy):
    if dst.exists():
        return
    if copy:
        shutil.copyfile(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--coco-dir', type=Path)
    p.add_argument('--download-coco', action='store_true')
    p.add_argument('--sop-dir', type=Path, required=True)
    p.add_argument('--out', type=Path, default=ROOT / 'resources' / 'images')
    p.add_argument('--multiview', action='store_true')
    p.add_argument('--copy', action='store_true', help='copy files instead of making symbolic links')
    args = p.parse_args()
    if not args.coco_dir and not args.download_coco:
        p.error('give --coco-dir or --download-coco')
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / '_coco'
    rows = read(ROOT / 'benchmark' / 'records.csv.gz')
    if args.multiview:
        rows += read(ROOT / 'benchmark' / 'multiview' / 'records.csv.gz')
    missing = 0
    for i, r in enumerate(rows):
        source = r['image_source']
        dst = args.out / f"{r['record_id']}.jpg"
        if r.get('source_dataset', 'stanford_online_products') == 'coco2017':
            if args.download_coco:
                src = cache / source
                if not src.exists():
                    src.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(COCO_URL.format(source), src)
            else:
                src = args.coco_dir / source
        else:
            src = args.sop_dir / source
        if not src.exists():
            missing += 1
            continue
        place(src, dst, args.copy)
        if (i + 1) % 5000 == 0:
            print(f'{i + 1} of {len(rows)} records', flush=True)
    print(f'placed {len(rows) - missing} images in {args.out}; {missing} missing')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
