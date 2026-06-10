"""
Модуль витягу 9 стилометричних ознак
"""

import math
import re
from collections import Counter

import numpy as np


FUNCTION_WORDS_EN = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "by",
    "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "and", "or",
    "but", "if", "while", "when", "where", "what", "who", "how", "not",
    "no", "so", "very", "just", "than", "also", "such",
}

FUNCTION_WORDS_UK = {
    "і", "й", "та", "а", "але", "або", "що", "як", "це", "той", "ця",
    "те", "ці", "на", "у", "в", "до", "від", "за", "під", "над", "перед",
    "між", "для", "з", "зі", "із", "без", "про", "при", "після", "коли",
    "де", "куди", "звідки", "хто", "який", "не", "ні", "так", "є", "був",
    "була", "було", "були", "буде", "ми", "ви", "я", "ти", "він", "вона",
    "воно", "вони", "його", "її", "свій", "також", "ще", "вже",
}

FEATURE_NAMES = [
    "yules_k", "honore_r", "mattr", "burstiness", "sent_start_div",
    "punct_entropy", "fw_deviation", "avg_sent_len", "ttr",
]


def detect_lang(text):
    cyr = sum(1 for ch in text if "\u0400" <= ch <= "\u04FF")
    lat = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    if cyr > lat:
        return "uk"
    if lat > 0:
        return "en"
    return "other"


def tokenize(text):
    return re.findall(r"[\w']+", text.lower(), flags=re.UNICODE)


def split_sentences(text):
    parts = re.split(r"[.!?]+\s+|[.!?]+$", text)
    return [s.strip() for s in parts if s.strip()]


def yules_k(tokens):
    n = len(tokens)
    if n < 2:
        return 0.0
    spectrum = Counter(Counter(tokens).values())
    s = sum(m * (i ** 2) for i, m in spectrum.items())
    return 10000 * (s - n) / (n * n)


def honore_r(tokens):
    n = len(tokens)
    if n < 2:
        return 0.0
    freq = Counter(tokens)
    v = len(freq)
    v1 = sum(1 for c in freq.values() if c == 1)
    if v == 0 or v1 >= v:
        return 0.0
    return 100.0 * math.log(n) / (1 - v1 / v)


def mattr(tokens, window=50):
    if len(tokens) < window:
        return len(set(tokens)) / max(len(tokens), 1)
    ratios = [
        len(set(tokens[i:i + window])) / window
        for i in range(len(tokens) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def burstiness(sentence_token_lists):
    if len(sentence_token_lists) < 2:
        return 0.0
    lens = np.array([len(t) for t in sentence_token_lists], dtype=float)
    mu, sigma = lens.mean(), lens.std()
    if mu + sigma == 0:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def sentence_start_diversity(sentences):
    starts = []
    for s in sentences:
        toks = tokenize(s)
        if toks:
            starts.append(toks[0])
    if not starts:
        return 0.0
    return len(set(starts)) / len(starts)


def punct_entropy(text):
    puncts = [c for c in text if c in '.,;:!?-—()"\'']
    if not puncts:
        return 0.0
    counts = Counter(puncts)
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def function_word_deviation(tokens, lang_words, baseline=0.40):
    if not tokens:
        return 0.0
    observed = sum(1 for t in tokens if t in lang_words) / len(tokens)
    return abs(observed - baseline)


def avg_sentence_length(sentence_token_lists):
    if not sentence_token_lists:
        return 0.0
    return float(np.mean([len(t) for t in sentence_token_lists]))


def type_token_ratio(tokens):
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def extract_features(text):
    """Повертає вектор з 9 ознак у порядку FEATURE_NAMES."""
    tokens = tokenize(text)
    sents = split_sentences(text)
    sent_tokens = [tokenize(s) for s in sents]
    lang = detect_lang(text)
    fw = FUNCTION_WORDS_UK if lang == "uk" else FUNCTION_WORDS_EN
    return [
        yules_k(tokens),
        honore_r(tokens),
        mattr(tokens),
        burstiness(sent_tokens),
        sentence_start_diversity(sents),
        punct_entropy(text),
        function_word_deviation(tokens, fw),
        avg_sentence_length(sent_tokens),
        type_token_ratio(tokens),
    ]
