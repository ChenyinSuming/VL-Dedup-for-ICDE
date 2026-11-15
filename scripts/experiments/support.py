"""Small helpers shared by the experiment scripts."""
import json
import math
import re
import time
import unicodedata


def normalize_text(text):
    normalized = unicodedata.normalize('NFKC', str(text)).lower()
    normalized = re.sub(r'[^\w\s]+', ' ', normalized)
    return re.sub(r'\s+', ' ', normalized).strip()


def repair_json_like_object(candidate):
    repaired = candidate.replace('\\_', '_')
    enum_pattern = re.compile(r':\s*(yes|no|none|uncertain|certain|maybe|unsure|high|medium|low)\s*([,}])', flags=re.IGNORECASE)
    repaired = enum_pattern.sub(lambda match: f': "{match.group(1).lower()}"{match.group(2)}', repaired)
    return re.sub(r':\s*(True|False)\s*([,}])', lambda match: f': {match.group(1).lower()}{match.group(2)}', repaired)


def extract_json_object(text):
    """Return the first JSON object in a model answer, or None."""
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return None
    raw = str(text).strip().replace('\\_', '_')
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fence.group(1)] if fence else []
    start, end = raw.find('{'), raw.rfind('}')
    if 0 <= start < end:
        candidates.append(raw[start:end + 1])
    candidates.append(raw)
    for candidate in candidates:
        for attempt in (candidate, repair_json_like_object(candidate)):
            try:
                parsed = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def cpu():
    return time.process_time()


def build_hnsw_edges(builder, embedding, channel, k, edges, partitions, config):
    """Add the edges of one embedding channel with an HNSW index and inner-product search."""
    before, measurements = len(edges), {}

    def measure(name, function):
        wall, process = time.perf_counter(), cpu()
        value = function()
        measurements[name] = dict(wall_seconds=time.perf_counter() - wall, cpu_seconds=cpu() - process)
        return value

    index = builder.faiss.IndexHNSWFlat(embedding.shape[1], config['hnsw_m'], builder.faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config['ef_construction']
    index.hnsw.efSearch = config['ef_search']
    builder.faiss.omp_set_num_threads(config['build_threads'])
    measure('index_add', lambda: index.add(embedding))
    builder.faiss.omp_set_num_threads(config['search_threads'])
    scores, indices = measure('search', lambda: index.search(embedding, k + 1))

    def merge():
        for i in range(len(embedding)):
            for score, j in zip(scores[i], indices[i]):
                j = int(j)
                if j >= 0 and j != i:
                    builder.update_edge(edges, i, j, channel, {channel + '_score': float(score)}, partitions)

    measure('python_union', merge)
    return len(edges) - before, measurements


def export_cluster(ids, representative, judgments):
    """Run the cluster export and name the flagged state `pending`, as the experiment scripts do."""
    from scripts.retained_set_helpers import export_cluster as export
    return {arm: dict(retained=value['retained'], pending=value['held'], deleted=value['deleted'])
            for arm, value in export(ids, representative, judgments).items()}
