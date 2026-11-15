"""COVERAGE_CHECK_RUN: frozen, text-only per-deletion coverage feasibility check."""
from scripts.vl_paths import RESOURCES, COCO_ARCHIVES, CONFIG
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "outputs/reports/coverage_check"
RUN = (RESOURCES / "runs/coverage_check")
GROUPS = ("noisy_pool_a", "retained_audit_a", "retained_audit_b")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def writel(path, values):
    with Path(path).open("x") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def protocol():
    return read(CONFIG / 'coverage_check.json')


def indexed(values, key):
    result = {v[key]: v for v in values}
    assert len(result) == len(values), ("duplicate", key)
    return result


def prepare():
    p = protocol()
    b = Path(p["input_contract"]["noisy_pool_run"])
    records = indexed(rows(b / "a_records.jsonl"), "record_id")
    exports = indexed(rows(b / "a_exports.jsonl"), "component")
    units = read(b / "a_audit_units.json")
    coverage = indexed(rows(b / "coverage_by_caption.jsonl"), "deleted_id")
    models = ["gpt-6.1-sol", "gpt-5.6-terra"]
    reasons = {}
    for path in sorted((b / "a_audit_2").glob("*_terminal.json")):
        terminal = read(path)
        assert terminal["status"] == "complete"
        for label in terminal["labels"]:
            reasons[(label["deleted_id"], terminal["model"])] = label
    inputs = []
    for image_id, image_units in units.items():
        for u in image_units:
            rid = u["deleted_id"]
            record = records[rid]
            export = exports[record["component"]]
            assert rid in export["deleted"] and record["sampled_image"]
            retained = [{"record_id": k, "caption": records[k]["caption"]} for k in export["retained"]]
            assert retained == u["retained"] and record["caption"] == u["deleted_caption"]
            c = coverage[rid]
            assert str(c["image_id"]) == image_id
            votes = [c["votes"][m] for m in models]
            assert all(reasons[(rid, m)]["covered"] is v for m, v in zip(models, votes))
            assert c["strict"] == all(v is True for v in votes)
            assert c["loose"] == any(v is True for v in votes)
            inputs.append(dict(group="noisy_pool_a", unit_id=rid, source_id=rid,
                               image_id=int(image_id), deleted_caption=u["deleted_caption"],
                               retained=retained, auditor_models=models, auditor_votes=votes,
                               auditor_raw_labels=votes,
                               auditor_reasons=[reasons[(rid, m)]["reason"] for m in models]))
    expected_noisy_pool = {r["record_id"] for r in records.values()
                    if r["sampled_image"] and r["record_id"] in exports[r["component"]]["deleted"]}
    assert {r["source_id"] for r in inputs} == set(coverage) == expected_noisy_pool
    parent = ROOT / p["input_contract"]["retained_audit_report"]
    frozen = read(p["input_contract"]["retained_audit_records"])
    records34 = indexed(frozen["records"], "record_id")
    cells = rows(parent / "audit_cells.jsonl")
    audits = [indexed(rows(parent / f"audit_{i}.jsonl"), "cell_id") for i in (1, 2)]
    agreement = indexed(rows(parent / "agreement.jsonl"), "cell_id")
    assert set(indexed(cells, "cell_id")) == set(agreement) == set(audits[0]) == set(audits[1])
    assert indexed(cells, "cell_id") == indexed(frozen["cells"], "cell_id")
    binary = {"covered_single": True, "covered_joint": True, "lost": False, "uncertain": None}
    for cell in cells:
        cid = cell["cell_id"]
        retained_ids = cell["retained_ids"]
        assert retained_ids == frozen["retained_ids"][cell["arm"]]
        a = [audit[cid] for audit in audits]
        for i, audit in enumerate(a, 1):
            assert audit["status"] == "completed"
            assert audit["source_id"] == cell["source_id"] and audit["retained_ids"] == retained_ids
            assert audit["audit"] == agreement[cid][f"audit_{i}"]["audit"]
        labels = [v["audit"]["label"] for v in a]
        inputs.append(dict(group="retained_audit_" + cell["strategy"], unit_id=cid,
                           source_id=cell["source_id"], cluster_id=cell["cluster_id"],
                           deleted_caption=records34[cell["source_id"]]["text"],
                           retained=[{"record_id": k, "caption": records34[k]["text"]} for k in retained_ids],
                           auditor_models=[v["model"] for v in a], auditor_raw_labels=labels,
                           auditor_votes=[binary[label] for label in labels],
                           auditor_reasons=[v["audit"]["reason"] for v in a]))
    inputs.sort(key=lambda v: (v["group"], v["unit_id"]))
    assert len({(v["group"], v["unit_id"]) for v in inputs}) == len(inputs)
    counts = {}
    for group in GROUPS:
        group_rows = [r for r in inputs if r["group"] == group]
        votes = [r["auditor_votes"] for r in group_rows]
        if group == "noisy_pool_a":
            counts[group] = dict(deletions=len(votes), strict_covered=sum(all(v is True for v in x) for x in votes),
                                 loose_covered=sum(any(v is True for v in x) for x in votes))
        else:
            counts[group] = dict(deletions=len(votes), strict_lost=sum(all(v is False for v in x) for x in votes),
                                 loose_lost=sum(any(v is False for v in x) for x in votes))
    passed = all(counts[g] == p["gates"][g] for g in GROUPS)
    dump(REPORT / "gate.json", dict(passed=passed, independently_computed=counts,
                                     retained_audit_source=p["input_contract"]["retained_audit_report"]))
    assert passed, counts
    writel(RUN / "inputs.jsonl", inputs)
    dump(RUN / "protocol.json", p)
    print(json.dumps(counts))


def parse_output(raw, finish_reason):
    if finish_reason != "stop":
        return dict(valid=False, covered=None, reason=None, error="finish_reason=" + str(finish_reason))
    value = raw.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    elif value.startswith("```\n") and value.endswith("\n```"):
        value = value[4:-4]
    try:
        data = json.loads(value)
        assert isinstance(data, dict) and set(data) == {"covered", "reason"}
        assert data["covered"] is None or type(data["covered"]) is bool
        assert isinstance(data["reason"], str) and data["reason"].strip()
    except (ValueError, AssertionError) as exc:
        return dict(valid=False, covered=None, reason=None, error="invalid JSON contract: " + str(exc))
    return dict(valid=True, covered=data["covered"], reason=data["reason"], error=None)


def infer():
    assert os.environ.get("VL_DEDUP_GPU_LEASE") == "1"
    assert read(REPORT / "gate.json")["passed"]
    assert not (RUN / "predictions.jsonl").exists()
    start = time.perf_counter()
    snapshot = subprocess.check_output(["nvidia-smi"], text=True)
    (RUN / "nvidia_smi_before_load.txt").write_text(snapshot)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True)
    assert not processes.strip(), "GPU occupied; model not loaded"
    print(snapshot, flush=True)
    p = protocol()
    sys.path.insert(0, p["model"]["dependency_site"])
    # vLLM's registry inspection launches python without -I in a child process.
    os.environ["PYTHONPATH"] = p["model"]["dependency_site"]
    # Relative UNIX socket addresses avoid sockaddr_un's 107-character limit.
    os.chdir(RUN)
    os.environ["VLLM_RPC_BASE_PATH"] = "."
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import importlib.metadata

    config = p["model"]
    inputs = rows(RUN / "inputs.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(config["path"], local_files_only=True)
    prompts = []
    for r in inputs:
        content = p["prompt_prefix"] + json.dumps({"deleted_caption": r["deleted_caption"],
                    "retained_captions": [v["caption"] for v in r["retained"]]}, ensure_ascii=False)
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
        assert tokens + p["decoding"]["max_tokens"] <= config["max_model_len"], (r["unit_id"], tokens)
        prompts.append(dict(group=r["group"], unit_id=r["unit_id"], content=content, rendered=rendered, prompt_tokens=tokens))
    writel(RUN / "prompts.jsonl", prompts)
    kwargs = {k: v for k, v in config.items() if k not in ("name", "path", "backend", "dependency_site")}
    llm = LLM(model=config["path"], **kwargs)
    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=p["decoding"]["max_tokens"], n=1)
    session = dict(started_utc=datetime.now(timezone.utc).isoformat(), requests_submitted=0, responses=0,
                   lock_mode="flock -n; nonblocking acquisition", lock_wait_seconds=0,
                   batches=0, invalid=0, abstentions=0, prompt_token_range=[min(r["prompt_tokens"] for r in prompts), max(r["prompt_tokens"] for r in prompts)],
                   versions={name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")})
    generation = time.perf_counter()
    try:
        with (RUN / "predictions.jsonl").open("x") as handle:
            size = p["decoding"]["batch_size"]
            for offset in range(0, len(prompts), size):
                batch = prompts[offset:offset + size]
                session["requests_submitted"] += len(batch)
                dump(RUN / "session.json", session)
                result = llm.generate([r["rendered"] for r in batch], sampling, use_tqdm=False)
                assert len(result) == len(batch)
                for r, out in zip(batch, result):
                    output = out.outputs[0]
                    parsed = parse_output(output.text, output.finish_reason)
                    prediction = {k: r[k] for k in ("group", "unit_id")}
                    prediction.update(raw=output.text, finish_reason=output.finish_reason,
                                      completion_tokens=len(output.token_ids), **parsed)
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    session["responses"] += 1
                    session["invalid"] += not parsed["valid"]
                    session["abstentions"] += parsed["valid"] and parsed["covered"] is None
                handle.flush()
                session["batches"] += 1
                print(json.dumps({k: session[k] for k in ("responses", "invalid", "abstentions")}), flush=True)
        session["complete"] = True
    finally:
        session.update(generation_wall_seconds=time.perf_counter() - generation,
                       session_wall_seconds=time.perf_counter() - start)
        dump(RUN / "session.json", session)


def divide(a, b):
    return a / b if b else None


def confusion(items, loose):
    c = Counter()
    for r in items:
        positive = (any if loose else all)(v is False for v in r["auditor_votes"])
        predicted = r["prediction"]["covered"] is False
        c[("t" if positive == predicted else "f") + ("p" if predicted else "n")] += 1
    return dict(n=len(items), tp=c["tp"], fp=c["fp"], fn=c["fn"], tn=c["tn"],
                precision=divide(c["tp"], c["tp"] + c["fp"]), recall=divide(c["tp"], c["tp"] + c["fn"]))


def group_metrics(items):
    resolved = [r for r in items if r["prediction"]["valid"] and r["prediction"]["covered"] is not None]
    paired = [r for r in resolved if None not in r["auditor_votes"]]
    consensus = [r for r in items if None not in r["auditor_votes"] and len(set(r["auditor_votes"])) == 1]
    evaluated = [r for r in paired if len(set(r["auditor_votes"])) == 1]
    n = len(evaluated)
    match = sum(r["prediction"]["covered"] == r["auditor_votes"][0] for r in evaluated)
    p_ref = divide(sum(r["auditor_votes"][0] is True for r in evaluated), n)
    p_model = divide(sum(r["prediction"]["covered"] is True for r in evaluated), n)
    expected = p_ref * p_model + (1 - p_ref) * (1 - p_model) if n else None
    accuracy = divide(match, n)
    kappa = (accuracy - expected) / (1 - expected) if expected is not None and expected != 1 else None
    def label(v):
        return "null" if v is None else str(v).lower()
    return dict(deletions=len(items), auditor_either_null=sum(None in r["auditor_votes"] for r in items),
                auditor_binary_disagreement=sum(None not in r["auditor_votes"] and len(set(r["auditor_votes"])) != 1 for r in items),
                auditor_raw_label_disagreement=sum(r["auditor_raw_labels"][0] != r["auditor_raw_labels"][1] for r in items),
                auditor_vote_pairs=dict(Counter("/".join(map(label, r["auditor_votes"])) for r in items)),
                auditor_raw_label_pairs=dict(Counter("/".join(map(str, r["auditor_raw_labels"])) for r in items)),
                model_invalid=sum(not r["prediction"]["valid"] for r in items),
                model_abstentions=sum(r["prediction"]["valid"] and r["prediction"]["covered"] is None for r in items),
                model_false=sum(r["prediction"]["covered"] is False for r in resolved), model_true=sum(r["prediction"]["covered"] is True for r in resolved),
                agreement=dict(auditor_consensus_units=len(consensus), evaluated=n, matches=match,
                               accuracy=accuracy, kappa=kappa, expected_agreement=expected,
                               all_consensus_match_fraction=divide(match, len(consensus)), excluded=len(consensus) - n),
                precision_recall=dict(primary_both_auditors_nonnull={"strict": confusion(paired, False), "loose": confusion(paired, True)},
                                      supplementary_observed_membership={"strict": confusion(resolved, False), "loose": confusion(resolved, True)},
                                      primary_excluded=len(items) - len(paired),
                                      primary_excluded_strict_positives=sum(all(v is False for v in r["auditor_votes"]) for r in items if r not in paired),
                                      primary_excluded_loose_positives=sum(any(v is False for v in r["auditor_votes"]) for r in items if r not in paired)))


def hypothetical(items):
    extra = [r for r in items if r["prediction"]["valid"] and r["prediction"]["covered"] is False]
    return dict(additional_retained=len(extra), unique_source_records=len({r["source_id"] for r in extra}),
                strict_lost=sum(all(v is False for v in r["auditor_votes"]) for r in extra),
                loose_lost=sum(any(v is False for v in r["auditor_votes"]) for r in extra),
                strict_covered=sum(all(v is True for v in r["auditor_votes"]) for r in extra),
                loose_covered=sum(any(v is True for v in r["auditor_votes"]) for r in extra),
                either_null=sum(None in r["auditor_votes"] for r in extra),
                vote_pairs=dict(Counter(str(r["auditor_votes"]) for r in extra)))


def analyze():
    inputs = rows(RUN / "inputs.jsonl")
    predictions = rows(RUN / "predictions.jsonl")
    pred = {(r["group"], r["unit_id"]): r for r in predictions}
    assert len(pred) == len(predictions) == len(inputs) == 860
    assert set(pred) == {(r["group"], r["unit_id"]) for r in inputs}
    session = read(RUN / "session.json")
    assert session["complete"] and session["responses"] == session["requests_submitted"] == 860
    for r in inputs:
        r["prediction"] = pred[(r["group"], r["unit_id"])]
    metrics = {g: group_metrics([r for r in inputs if r["group"] == g]) for g in GROUPS}
    effects = {g: hypothetical([r for r in inputs if r["group"] == g]) for g in ("noisy_pool_a", "retained_audit_b")}
    dump(REPORT / "results.json", dict(run_id=protocol()['run_id'],
         protocol_commit=subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(CONFIG / 'coverage_check.json')], text=True).strip(),
         gates=read(REPORT / "gate.json"), execution=session,
         startup_failures=[read(path) for path in sorted(RUN.glob("startup_failure_*.json"))],
         wall_clock_scope="Successful initialization+inference session; three failed pre-generation startups not timed cumulatively, raw logs preserved. Generation requests never retried.",
         groups=metrics, hypothetical=effects,
         auditor_input_forms={"noisy_pool_a": "text only", "retained_audit_a": "text plus images", "retained_audit_b": "text plus images"}))
    print(json.dumps(dict(metrics=metrics, hypothetical=effects), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "infer", "analyze"))
    args = parser.parse_args()
    {"prepare": prepare, "infer": infer, "analyze": analyze}[args.stage]()
