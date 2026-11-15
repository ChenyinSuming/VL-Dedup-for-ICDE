from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.vl_paths import RESOURCES
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
MODEL_CHOICES = ('phash', 'sscd', 'dinov2', 'clip')
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
STAGE_SEMANTIC_OUTPUTS = {'phash': ('phash.parquet', 'metadata.parquet'), 'sscd': ('image_emb.npy', 'metadata.parquet'), 'dinov2': ('image_emb.npy', 'metadata.parquet'), 'clip': ('image_emb.npy', 'text_emb.npy', 'clipscore.npy', 'metadata.parquet')}

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', type=Path, default=RESOURCES / 'datasets/VL-DedupBench-25K_release', help='Release package root used to resolve relative image paths.')
    parser.add_argument('--manifest', type=Path, default=repo_root() / 'data' / 'manifests' / 'vl_dedupbench_50k_manifest.parquet', help='Standardized manifest parquet.')
    parser.add_argument('--out-root', type=Path, default=repo_root() / 'features', help='Root directory for feature outputs.')
    parser.add_argument('--models', type=str, default=','.join(MODEL_CHOICES), help=f'Comma-separated subset of {MODEL_CHOICES}.')
    parser.add_argument('--limit', type=int, default=None, help='Optional row limit for smoke tests.')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for neural models.')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers for image models.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Torch device.')
    parser.add_argument('--sscd-path', type=Path, default=repo_root() / 'models' / 'sscd' / 'sscd_disc_mixup.torchscript.pt', help='Local SSCD TorchScript checkpoint.')
    parser.add_argument('--torchhub-dir', type=Path, default=repo_root() / 'models' / 'torchhub' / 'hub', help='Local Torch Hub directory containing DINOv2.')
    parser.add_argument('--dinov2-repo', type=Path, default=repo_root() / 'models' / 'torchhub' / 'hub' / 'facebookresearch_dinov2_main', help='Local DINOv2 torchhub repository.')
    parser.add_argument('--dinov2-model', type=str, default='dinov2_vitl14')
    parser.add_argument('--clip-model', type=str, default='openai/clip-vit-large-patch14-336', help='Hugging Face CLIP model ID or local model directory.')
    return parser.parse_args()

def selected_models(raw: str) -> list[str]:
    models = [item.strip().lower() for item in raw.split(',') if item.strip()]
    unknown = sorted(set(models) - set(MODEL_CHOICES))
    if unknown:
        raise ValueError(f'Unknown models: {unknown}. Valid choices: {MODEL_CHOICES}')
    return models

def resolve_image_path(release_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return release_dir / path

def load_rgb_image(path: Path, fallback_size: int) -> tuple[Image.Image, bool, str]:
    try:
        image = Image.open(path).convert('RGB')
        return (image, True, '')
    except Exception as exc:
        return (Image.new('RGB', (fallback_size, fallback_size), color=(0, 0, 0)), False, str(exc))

def l2_normalize(array: np.ndarray, eps: float=1e-12) -> np.ndarray:
    norm = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norm, eps)

def base_metadata(manifest: pd.DataFrame, model_name: str, embedding_dim: int | None) -> pd.DataFrame:
    return pd.DataFrame({'record_id': manifest['record_id'].to_numpy(), 'row_index': np.arange(len(manifest), dtype=np.int64), 'source_model': model_name, 'embedding_dim': embedding_dim, 'normalized': embedding_dim is not None})

def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

class ImageTensorDataset(Dataset):

    def __init__(self, manifest: pd.DataFrame, release_dir: Path, transform: transforms.Compose, fallback_size: int) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.release_dir = release_dir
        self.transform = transform
        self.fallback_size = fallback_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, bool]:
        image_path = resolve_image_path(self.release_dir, self.manifest.at[index, 'image_path'])
        (image, read_ok, _) = load_rgb_image(image_path, self.fallback_size)
        return (self.transform(image), read_ok)

def image_loader(manifest: pd.DataFrame, release_dir: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    transform = transforms.Compose([transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC), transforms.CenterCrop(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    dataset = ImageTensorDataset(manifest, release_dir, transform, image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

def save_embedding_feature(out_dir: Path, manifest: pd.DataFrame, model_name: str, array: np.ndarray, read_ok: list[bool], filename: str='image_emb.npy', provenance: dict | None=None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / filename, array.astype(np.float32, copy=False))
    metadata = base_metadata(manifest, model_name, array.shape[1])
    metadata['read_ok'] = read_ok
    metadata['embedding_path'] = filename
    metadata.to_parquet(out_dir / 'metadata.parquet', index=False)
    write_json(out_dir / 'feature_info.json', {'model_name': model_name, 'embedding_path': filename, 'records': int(array.shape[0]), 'embedding_dim': int(array.shape[1]), 'normalized': True, 'dtype': 'float32', **(provenance or {})})

def extract_phash(manifest: pd.DataFrame, release_dir: Path, out_root: Path) -> None:
    import imagehash
    out_dir = out_root / 'phash'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in tqdm(manifest[['record_id', 'image_path']].itertuples(index=False), total=len(manifest), desc='Extracting pHash'):
        image_file = resolve_image_path(release_dir, row.image_path)
        (image, read_ok, error) = load_rgb_image(image_file, 224)
        if read_ok:
            phash = imagehash.phash(image)
            phash_hex = str(phash)
            (width, height) = image.size
        else:
            phash_hex = None
            width = None
            height = None
        rows.append({'record_id': row.record_id, 'row_index': len(rows), 'image_path': row.image_path, 'phash_hex': phash_hex, 'phash_int': int(phash_hex, 16) if phash_hex is not None else None, 'width': width, 'height': height, 'read_ok': read_ok, 'error': error})
        image.close()
    pd.DataFrame(rows).to_parquet(out_dir / 'phash.parquet', index=False)
    metadata = base_metadata(manifest, 'phash', None)
    metadata['read_ok'] = [row['read_ok'] for row in rows]
    metadata['embedding_path'] = 'phash.parquet'
    metadata.to_parquet(out_dir / 'metadata.parquet', index=False)
    write_json(out_dir / 'feature_info.json', {'model_name': 'phash', 'records': int(len(rows)), 'embedding_path': 'phash.parquet', 'normalized': False})

@torch.inference_mode()
def extract_sscd(manifest: pd.DataFrame, release_dir: Path, out_root: Path, checkpoint: Path, device: str, batch_size: int, num_workers: int) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(f'Missing SSCD checkpoint: {checkpoint}')
    model = torch.jit.load(str(checkpoint), map_location=device).eval()
    loader = image_loader(manifest, release_dir, image_size=288, batch_size=batch_size, num_workers=num_workers)
    batches: list[np.ndarray] = []
    read_ok: list[bool] = []
    for (images, ok) in tqdm(loader, total=len(loader), desc='Extracting SSCD'):
        images = images.to(device, non_blocking=True)
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = torch.nn.functional.normalize(output.float(), dim=1)
        batches.append(output.cpu().numpy())
        read_ok.extend([bool(item) for item in ok])
    embeddings = np.concatenate(batches, axis=0)
    save_embedding_feature(out_root / 'sscd', manifest, 'sscd_disc_mixup', embeddings, read_ok, provenance={'model_path': str(checkpoint)})

@torch.inference_mode()
def extract_dinov2(manifest: pd.DataFrame, release_dir: Path, out_root: Path, torchhub_dir: Path, dinov2_repo: Path, dinov2_model: str, device: str, batch_size: int, num_workers: int) -> None:
    if not dinov2_repo.exists():
        raise FileNotFoundError(f'Missing local DINOv2 repo: {dinov2_repo}')
    torch.hub.set_dir(str(torchhub_dir))
    model = torch.hub.load(str(dinov2_repo), dinov2_model, source='local').to(device).eval()
    loader = image_loader(manifest, release_dir, image_size=518, batch_size=batch_size, num_workers=num_workers)
    batches: list[np.ndarray] = []
    read_ok: list[bool] = []
    for (images, ok) in tqdm(loader, total=len(loader), desc='Extracting DINOv2'):
        images = images.to(device, non_blocking=True)
        output = model(images)
        if isinstance(output, dict):
            output = output.get('x_norm_clstoken', next(iter(output.values())))
        elif isinstance(output, (tuple, list)):
            output = output[0]
        output = torch.nn.functional.normalize(output.float(), dim=1)
        batches.append(output.cpu().numpy())
        read_ok.extend([bool(item) for item in ok])
    embeddings = np.concatenate(batches, axis=0)
    save_embedding_feature(out_root / 'dinov2', manifest, dinov2_model, embeddings, read_ok, provenance={'model_path': str(dinov2_repo)})

def load_clip_images(manifest_batch: pd.DataFrame, release_dir: Path) -> tuple[list[Image.Image], list[bool], list[str]]:
    images: list[Image.Image] = []
    read_ok: list[bool] = []
    errors: list[str] = []
    for image_path in manifest_batch['image_path']:
        (image, ok, error) = load_rgb_image(resolve_image_path(release_dir, image_path), 336)
        images.append(image)
        read_ok.append(ok)
        errors.append(error)
    return (images, read_ok, errors)

def clip_feature_tensor(output: object) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    pooler_output = getattr(output, 'pooler_output', None)
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f'Unsupported CLIP feature output type: {type(output)!r}')

@torch.inference_mode()
def extract_clip(manifest: pd.DataFrame, release_dir: Path, out_root: Path, clip_model: str, device: str, batch_size: int) -> None:
    from transformers import CLIPModel, CLIPProcessor
    out_dir = out_root / 'clip'
    out_dir.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(str(clip_model)).to(device).eval()
    processor = CLIPProcessor.from_pretrained(str(clip_model))
    clip_text_max_length = int(model.config.text_config.max_position_embeddings)
    image_batches: list[np.ndarray] = []
    text_batches: list[np.ndarray] = []
    read_ok: list[bool] = []
    errors: list[str] = []
    for start in tqdm(range(0, len(manifest), batch_size), desc='Extracting CLIP'):
        batch = manifest.iloc[start:start + batch_size]
        (images, batch_ok, batch_errors) = load_clip_images(batch, release_dir)
        texts = batch['text'].astype(str).tolist()
        image_inputs = processor(images=images, return_tensors='pt')
        text_inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True, max_length=clip_text_max_length)
        image_inputs = {key: value.to(device) for (key, value) in image_inputs.items()}
        text_inputs = {key: value.to(device) for (key, value) in text_inputs.items()}
        image_emb = clip_feature_tensor(model.get_image_features(**image_inputs)).float()
        text_emb = clip_feature_tensor(model.get_text_features(**text_inputs)).float()
        image_emb = torch.nn.functional.normalize(image_emb, dim=1)
        text_emb = torch.nn.functional.normalize(text_emb, dim=1)
        image_batches.append(image_emb.cpu().numpy())
        text_batches.append(text_emb.cpu().numpy())
        read_ok.extend(batch_ok)
        errors.extend(batch_errors)
        for image in images:
            image.close()
    image_array = np.concatenate(image_batches, axis=0).astype(np.float32, copy=False)
    text_array = np.concatenate(text_batches, axis=0).astype(np.float32, copy=False)
    clipscore = np.sum(image_array * text_array, axis=1).astype(np.float32, copy=False)
    np.save(out_dir / 'image_emb.npy', image_array)
    np.save(out_dir / 'text_emb.npy', text_array)
    np.save(out_dir / 'clipscore.npy', clipscore)
    metadata = base_metadata(manifest, 'openai_clip_vit_large_patch14_336', image_array.shape[1])
    metadata['read_ok'] = read_ok
    metadata['error'] = errors
    metadata['image_embedding_path'] = 'image_emb.npy'
    metadata['text_embedding_path'] = 'text_emb.npy'
    metadata['clipscore_path'] = 'clipscore.npy'
    metadata.to_parquet(out_dir / 'metadata.parquet', index=False)
    write_json(out_dir / 'feature_info.json', {'model_name': 'openai/clip-vit-large-patch14-336', 'model_path': str(clip_model), 'records': int(len(manifest)), 'image_embedding_dim': int(image_array.shape[1]), 'text_embedding_dim': int(text_array.shape[1]), 'normalized': True, 'dtype': 'float32'})

def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = sorted({'record_id', 'image_path', 'text'} - set(manifest.columns))
    if missing:
        raise ValueError(f'Manifest missing required columns: {missing}')

def main() -> None:
    args = parse_args()
    models = selected_models(args.models)
    manifest = pd.read_parquet(args.manifest)
    validate_manifest(manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    release_dir = args.release_dir.resolve()
    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    if 'phash' in models:
        extract_phash(manifest, release_dir, out_root)
    if 'sscd' in models:
        extract_sscd(manifest, release_dir, out_root, args.sscd_path.resolve(), args.device, args.batch_size, args.num_workers)
    if 'dinov2' in models:
        extract_dinov2(manifest, release_dir, out_root, args.torchhub_dir.resolve(), args.dinov2_repo.resolve(), args.dinov2_model, args.device, args.batch_size, args.num_workers)
    if 'clip' in models:
        extract_clip(manifest, release_dir, out_root, args.clip_model, args.device, args.batch_size)
if __name__ == '__main__':
    main()
