# VL-Dedup and VL-DedupBench

Code, data and results for the paper
**"VL-Dedup: Graded Multimodal Entity Resolution for Image-Text Data Deduplication"**.

VL-Dedup removes duplicate image-text records from a corpus. It retrieves candidate pairs
(Stages A–B), scores them and builds visual-entity clusters of at most K records (Stages C–D),
checks the records inside each cluster with a vision-language model (Stage E), and exports the
retained set with caption and mismatch actions (Stage F).

VL-DedupBench has 50,000 records from COCO and Stanford Online Products (SOP) and 66,788 labeled
pairs, with test surfaces for edited images, multi-view products and larger entities.

## Repository structure

```
benchmark/      VL-DedupBench manifests
artifacts/      trained models, cluster labels, VLM outputs, exports and audit labels
results/        result files behind the paper tables and figures
scripts/        pipeline, evaluation and experiment scripts
tests/          tests that run without images or GPUs
environment/    package versions
```

## Installation

```bash
pip install -r environment/requirements.txt          # full pipeline
pip install -r environment/requirements-check.txt    # checks and tests only
```

## Quick start

Reproduce the released results without a GPU or images:

```bash
python scripts/recompute_graph_metrics.py   # Table IV from the released cluster labels
python -m scripts.retained_set_export       # the retained-set export
python scripts/recompute_stage_cd.py        # Stages C–D from the released models
python -m pytest tests -q
```

## Data

`benchmark/records.csv.gz` lists the records and `benchmark/pairs.csv.gz` the labeled pairs.
Each pair has a `grade` (L0–L3, M, HN), a `relation` and a `visual_relation`. The test surfaces are
in `benchmark/edit_surfaces/`, `benchmark/multiview/` and `benchmark/larger_entities/`, and the
full corpora of Table VIII are in `benchmark/full_corpora/`.

The repository contains no images. Download COCO 2017 and SOP, then run

```bash
python scripts/prepare_images.py --coco-dir COCO_DIR --sop-dir SOP_DIR --out resources/images
```

## Running the pipeline

Scripts read inputs from `resources/` and write results to `outputs/reports/`
(see `scripts/vl_paths.py`).

| Stage | Scripts |
|---|---|
| Test surfaces | `prepare_images.py`, `build_edit_stress.py`, `build_large_entities.py` |
| A–B | `extract_features.py`, `build_candidates.py`, `pair_features.py` |
| C–D | `train_visual_classifier.py` |
| E | `build_cluster_pairs.py`, `run_stage_e_clusters.py`, `score_caption_mismatches.py` |
| F | `run_caption_actions.py`, `run_combined_actions.py`, `retained_set_export.py` |

`scripts/experiments/` holds the remaining experiments, with settings in
`scripts/experiments/config/`.

## Paper results

| Paper element | Result file |
|---|---|
| Table III | `results/benchmark.csv`, `results/relations.csv` |
| Table IV | `results/graph.csv`, `results/graph_labeled_pairs.csv` |
| Table V | `results/ablation.csv` |
| Table VI | `results/mismatch.csv`, `results/vlm_direct_mismatch.csv` |
| Table VII | `results/clustering.csv` |
| Table VIII | `results/scale*.csv` |
| Table IX, Figure 7 | `results/downstream*.csv` |
| Figure 4 | `results/k_curve_*.csv`, `results/k_rules*.csv` |
| Figure 5 | `results/retrieval_scaling.csv`, `results/cost.csv` |
| Figure 6 | `results/retained_set_*.csv` |

`results/configuration.json` holds the features and the values of τ, K and t chosen on validation.

## License

Code is released under the MIT License (`LICENSE`). Data in `benchmark/`, `artifacts/` and
`results/` is released under CC BY 4.0 (`LICENSE-DATA`). COCO and SOP images keep the terms of
their sources.

## Citation

See `CITATION.cff`.
