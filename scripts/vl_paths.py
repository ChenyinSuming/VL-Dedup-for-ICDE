import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
RESOURCES = Path(os.environ.get('VL_DEDUP_RESOURCES', REPO_ROOT / 'resources'))
REPORTS = Path(os.environ.get('VL_DEDUP_REPORTS', REPO_ROOT / 'outputs' / 'reports'))
ANALYSIS = REPORTS / 'analysis'
COCO_ARCHIVES = Path(os.environ.get('VL_DEDUP_COCO_ARCHIVES', RESOURCES / 'coco2017_archives'))
CONFIG = REPO_ROOT / 'scripts' / 'experiments' / 'config'
