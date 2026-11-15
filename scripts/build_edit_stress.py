import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import RESOURCES, REPORTS
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import time
import pandas as pd
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
RESOURCES = RESOURCES
SOURCE = RESOURCES / 'runs/pair_features'
OUTPUT = RESOURCES / 'runs/edit_stress'
REPORT = Path(__file__).resolve().parents[1] / str(REPORTS / 'edit_stress')
MODELS = ('logistic', 'hist_gradient_boosting', 'mlp')
LEVELS = ('light', 'mild', 'strong')
# Seed offset of each edit level. The offsets are fixed so that every image keeps the seed it was
# rendered with; light is the edit that Mixed applies to half of the images.
SEED_OFFSET = {'light': 0, 'mild': 1, 'strong': 3}
SEED = 20_261_026
WATERMARK_TEXT = 'R24 EDIT COPY'

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def parameters(level, seed, flipped=False):
    rng = random.Random(seed)
    return dict(crop_fraction=dict(light=0.0, mild=0.12, strong=0.35)[level], crop_anchor='center', scale=0.9 if level == 'light' else 1.0, quality=dict(light=85, mild=70, strong=70)[level], brightness=rng.choice([0.92, 1.08]) if level == 'mild' else 1.0, rotation_degrees=dict(light=0, mild=0, strong=6)[level], gaussian_sigma=0.0, flipped=bool(flipped and level == 'strong'), color_multipliers=[rng.uniform(0.95, 1.05) for _ in range(3)] if level == 'strong' else [1.0, 1.0, 1.0], watermark=WATERMARK_TEXT if level == 'strong' else None)

def transform(image, params):
    image = ImageOps.exif_transpose(image).convert('RGB')
    (w, h) = image.size
    fraction = params['crop_fraction']
    (cw, ch) = (max(1, round(w * (1 - fraction))), max(1, round(h * (1 - fraction))))
    (left, top) = ((w - cw) // 2, (h - ch) // 2)
    box = (left, top, left + cw, top + ch)
    image = image.crop(box)
    if params['scale'] != 1:
        image = image.resize(tuple((max(1, round(x * params['scale'])) for x in image.size)), Image.Resampling.LANCZOS)
    if params['rotation_degrees']:
        image = image.rotate(params['rotation_degrees'], Image.Resampling.BICUBIC, expand=True, fillcolor=(127, 127, 127))
    if params['gaussian_sigma']:
        image = image.filter(ImageFilter.GaussianBlur(params['gaussian_sigma']))
    if params['flipped']:
        image = ImageOps.mirror(image)
    if params['brightness'] != 1:
        image = ImageEnhance.Brightness(image).enhance(params['brightness'])
    image = Image.merge('RGB', tuple((c.point(lambda x, m=m: min(255, max(0, round(x * m)))) for (c, m) in zip(image.split(), params['color_multipliers']))))
    if params['watermark']:
        draw = ImageDraw.Draw(image, 'RGBA')
        font_size = max(10, min(22, image.width // 18))
        xy = (max(1, image.width // 40), max(1, image.height - font_size * 2))
        draw.text(xy, params['watermark'], fill=(255, 255, 255, 140), stroke_width=1, stroke_fill=(0, 0, 0, 110), font_size=font_size)
    return (image, box)

def freeze(output=OUTPUT, report=REPORT):
    frame = pd.read_parquet(SOURCE / 'public_manifest.parquet')
    pairs = pd.read_parquet(SOURCE / 'full_test_predictions.parquet', columns=['pair_id', 'record_id_a', 'record_id_b'])
    ids = sorted(set(pairs.record_id_a) | set(pairs.record_id_b))
    selected = frame.set_index('record_id').loc[ids].reset_index()
    if len(frame) != 50000 or len(ids) != 5709 or len(pairs) != 10037:
        raise ValueError('Unexpected fixed input identity')
    native = pd.read_json(RESOURCES / 'datasets/benchmark/public_records.jsonl', lines=True)
    native = native.set_index('record_id').loc[frame.record_id]
    if native.text.tolist() != frame.text.tolist() or [r[0]['image_path'] for r in native.images] != frame.image_path.tolist():
        raise ValueError('benchmark caption cache differs')
    output.mkdir(parents=True, exist_ok=True)
    report.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / 'membership.csv', index=False)
    pairs.to_parquet(output / 'test_identity.parquet', index=False)
    rng = random.Random(SEED)
    flip_ids = set(rng.sample(ids, len(ids) // 2))
    rows = []
    for level in LEVELS:
        for (i, record) in enumerate(selected.itertuples()):
            seed = SEED + 100000 * SEED_OFFSET[level] + i
            path = output / 'edited_images' / level / (record.record_id + '.jpg')
            params = parameters(level, seed, record.record_id in flip_ids)
            rows.append(dict(record_id=record.record_id, image_id=record.record_id, original_path=record.image_path, new_path=str(path), level=level, seed=seed, parameters=json.dumps(params, sort_keys=True), flipped=params['flipped'], edit_status='not_rendered', subject_validity='preserved_no_crop' if level == 'light' else 'unreviewed_crop', invalid_edit=False if level == 'light' else '', error=''))
    pd.DataFrame(rows).to_csv(report / 'stress_manifest.csv', index=False)
    settings = dict(seed=SEED, levels=LEVELS, records=len(frame), edited_records_per_level=len(ids), source=str(SOURCE))
    dump(output / 'settings.json', settings)
    return settings

def render_row(row):
    path = Path(row['new_path'])
    params = json.loads(row['parameters'])
    try:
        if path.exists():
            raise FileExistsError('Refusing to overwrite an unreceipted image')
        with Image.open(row['original_path']) as image:
            (result, box) = transform(image, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        result.save(path, format='JPEG', quality=params['quality'])
        row.update(edit_status='completed', crop_box=json.dumps(box), output_width=result.width, output_height=result.height, bytes=path.stat().st_size)
    except Exception as error:
        row.update(edit_status='failed', error=f'{type(error).__name__}: {error}')
    return row

def generate(level, output=OUTPUT, report=REPORT):
    manifest = pd.read_csv(report / 'stress_manifest.csv', keep_default_na=False)
    rows = manifest.loc[manifest.level == level].to_dict('records')
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        done = list(executor.map(render_row, rows))
    other = manifest.loc[manifest.level != level].to_dict('records')
    pd.DataFrame(other + done).sort_values(['level', 'record_id']).to_csv(report / 'stress_manifest.csv', index=False)
    result = dict(level=level, status='completed' if all((r['edit_status'] == 'completed' for r in done)) else 'failed', completed=sum((r['edit_status'] == 'completed' for r in done)), failed=sum((r['edit_status'] == 'failed' for r in done)), wall_seconds=time.perf_counter() - started, invalid_edit_confirmed=0, subject_review_pending=sum((r['subject_validity'] == 'unreviewed_crop' for r in done)))
    dump(output / level / 'render_summary.json', result)
    if result['failed']:
        raise RuntimeError('Image render failures; see manifest, do not retry silently')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'generate'])
    p.add_argument('--level', choices=LEVELS)
    args = p.parse_args()
    if args.stage == 'freeze':
        result = freeze()
    else:
        if not args.level:
            p.error('generate requires --level')
        result = generate(args.level)
    print(json.dumps(result, ensure_ascii=False))
if __name__ == '__main__':
    main()
