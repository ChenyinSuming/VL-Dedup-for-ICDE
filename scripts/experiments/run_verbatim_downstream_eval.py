#!/usr/bin/env python3
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
"""Use the frozen official evaluator with PTB/SPICE scratch paths in VERBATIM_EXPORT_RUN."""
from pathlib import Path
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.experiments.run_verbatim_export import RUN


def local_scratch():
    import pycocoevalcap.tokenizer.ptbtokenizer as ptb
    import pycocoevalcap.spice.spice as spice
    for module,folder,assets in [(ptb,'tokenizer',['stanford-corenlp-3.4.1.jar']),
                                (spice,'spice',['spice-1.0.jar','lib'])]:
        original=Path(module.__file__)
        target=RUN/'eval_runtime'/folder
        target.mkdir(parents=True,exist_ok=True)
        for name in assets:
            source=original.parent/name
            if not source.exists():raise RuntimeError(f'Frozen evaluator asset missing: {source}')
            link=target/name
            if not link.exists():link.symlink_to(source)
        module.__file__=str(target/original.name)


if __name__=='__main__':
    local_scratch()
    from scripts.experiments.run_caption_downstream_eval import main
    main()
