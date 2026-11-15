import re
import unicodedata
SHINGLE_K = 3

def normalize_baseline_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', str(text)).lower()
    normalized = ''.join((' ' if unicodedata.category(character).startswith('P') else character for character in normalized))
    return re.sub('\\s+', ' ', normalized).strip()

def _word_shingles(text: str) -> frozenset[tuple[str, ...]]:
    words = normalize_baseline_text(text).split()
    if not words:
        return frozenset()
    if len(words) < SHINGLE_K:
        return frozenset({tuple(words)})
    return frozenset((tuple(words[start:start + SHINGLE_K]) for start in range(len(words) - SHINGLE_K + 1)))
