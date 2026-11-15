"""Provision CAPTION_DOWNSTREAM_RUN-only dependencies and official pretrained/evaluation resources."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True

RES = RESOURCES
RUN = RES / 'runs/caption_downstream'
NOISY_POOL_RUN = RES / 'runs/caption_pilot'
ROOT = Path(__file__).resolve().parents[1]
ENV = RUN / 'caption-env'


def dump(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(args, env):
    print(json.dumps({'command': args}), flush=True)
    subprocess.run(args, check=True, env=env)


def download(item):
    import requests
    url, destination = item
    destination.parent.mkdir(parents=True, exist_ok=True)
    marker = destination.with_suffix(destination.suffix + '.download.json')
    if marker.exists():
        return json.loads(marker.read_text())
    temporary = destination.with_suffix(destination.suffix + '.partial')
    offset = temporary.stat().st_size if temporary.exists() else 0
    with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                      stream=True, timeout=(60, 600)) as response:
        response.raise_for_status()
        append = offset > 0 and response.status_code == 206
        with temporary.open('ab' if append else 'wb') as stream:
            for block in response.iter_content(4*1024*1024):
                stream.write(block)
    temporary.replace(destination)
    receipt = dict(url=url, path=str(destination), bytes=destination.stat().st_size)
    dump(marker, receipt)
    return receipt


def main():
    assert json.loads((RUN / 'stage_e_timing_gate.json').read_text())['status'] == 'passed'
    if (RUN / 'setup_complete.json').exists():
        print((RUN / 'setup_complete.json').read_text(), flush=True)
        return
    started = time.monotonic()
    env = os.environ.copy()
    env.update(CONDA_PKGS_DIRS=str(RUN / 'conda_pkgs'),
               CONDA_ENVS_PATH=str(RUN / 'conda_envs'),
               CONDARC=str(RUN / 'condarc'),
               XDG_CACHE_HOME=str(RUN / 'runtime_cache'),
               PIP_CACHE_DIR=str(RUN / 'pip_cache'),
               PYTHONDONTWRITEBYTECODE='1',
               NLTK_DATA=str(RUN / 'nltk_data'))
    (RUN / 'condarc').write_text('channels:\n  - conda-forge\nshow_channel_urls: false\n')
    dump(RUN / 'setup_progress.json', dict(status='conda', pid=os.getpid()))
    if not (ENV / 'bin/java').exists():
        # Redirect conda's environment registry as well as its package/cache paths.
        # This changes only the provisioning process, never the base conda files.
        wrapper = RUN / 'conda_create.py'
        wrapper.write_text("from conda.core import envs_manager\n"
            f"envs_manager.get_user_environments_txt_file = lambda userhome='~': {str(RUN / 'conda_environments.txt')!r}\n"
            "from conda.cli.main import main\nraise SystemExit(main())\n")
        command([os.environ.get('VL_DEDUP_CONDA_PYTHON', sys.executable), str(wrapper), 'create', '-y',
                 '--prefix', str(ENV), '--override-channels', '-c', 'conda-forge',
                 'python=3.10', 'openjdk=8', 'pip'], env)
    python = str(ENV / 'bin/python')
    site = ENV / 'lib/python3.10/site-packages'
    # Existing GPU packages are consumed read-only; every new package installs here.
    (site / 'caption_downstream_existing_gpu_packages.pth').write_text(str(RES / 'envs/vl-dedup/lib/python3.10/site-packages') + '\n')
    dump(RUN / 'setup_progress.json', dict(status='python_dependencies', pid=os.getpid()))
    command([python, '-I', '-m', 'pip', 'install', '--no-deps', 'pycocoevalcap==1.2', 'nltk==3.9.1'], env)
    for name, url, branch in [('upstream_clip', 'https://github.com/openai/CLIP.git', None),
                              ('upstream_pattern', 'https://github.com/clips/pattern.git', '3.7-beta')]:
        destination = RUN / name
        if not destination.exists():
            args = ['git', 'clone', '--depth', '1']
            if branch:
                args += ['--branch', branch]
            command(args + [url, str(destination)], env)
        command([python, '-I', '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', str(destination)], env)
    models = RUN / 'caption_models'
    gpt = models / 'gpt2'
    gpt.mkdir(parents=True, exist_ok=True)
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.json', 'merges.txt']:
        source = NOISY_POOL_RUN / 'caption_models/gpt2' / name
        if source.exists() and not (gpt / name).exists():
            shutil.copyfile(source, gpt / name)
    dump(RUN / 'setup_progress.json', dict(status='official_downloads', pid=os.getpid()))
    downloads = [
        ('https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors', gpt / 'model.safetensors'),
        ('https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt', models / 'ViT-B-32.pt'),
        ('https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip', RUN / 'downloads/annotations_trainval2014.zip')]
    for name in ['stanford-corenlp-3.6.0.jar', 'stanford-corenlp-3.6.0-models.jar']:
        downloads.append(('https://repo.maven.apache.org/maven2/edu/stanford/nlp/stanford-corenlp/3.6.0/' + name,
                          site / 'pycocoevalcap/spice/lib' / name))
    with ThreadPoolExecutor(max_workers=4) as executor:
        transfers = list(executor.map(download, downloads))
    annotations = RUN / 'coco_annotations'
    annotations.mkdir(exist_ok=True)
    with zipfile.ZipFile(RUN / 'downloads/annotations_trainval2014.zip') as archive:
        for kind in ['captions', 'instances']:
            for split in ['train', 'val']:
                name = f'{kind}_{split}2014.json'
                with archive.open('annotations/' + name) as source, (annotations / name).open('wb') as target:
                    shutil.copyfileobj(source, target)
    command([python, '-I', '-m', 'nltk.downloader', '-d', str(RUN / 'nltk_data'),
             'punkt', 'punkt_tab', 'wordnet', 'omw-1.4', 'wordnet_ic', 'sentiwordnet'], env)
    with zipfile.ZipFile(RUN / 'nltk_data/corpora/wordnet.zip') as archive:
        archive.extractall(RUN / 'nltk_data/corpora')
    dump(RUN / 'setup_complete.json', dict(status='complete', transfers=transfers,
         environment=str(ENV), models=str(models), annotations=str(annotations),
         nltk_data=str(RUN / 'nltk_data'),
         pretrained_only=True, wall_seconds=time.monotonic()-started))
    dump(RUN / 'setup_progress.json', dict(status='complete', pid=os.getpid()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        dump(RUN / 'setup_progress.json', dict(status='failed', error=type(exc).__name__, detail=str(exc)))
        raise
