from __future__ import annotations

import math
import time
from collections import Counter, deque
from typing import Deque, Dict, Iterable, List, Sequence, Tuple


DEFAULT_TEXT_CORPUS = [
    "hello",
    "how are you",
    "i am fine",
    "thank you",
    "yes",
    "no",
    "you are fine",
    "i am okay",
    "thank you very much",
    "hello how are you",
]


class BigramLanguageModel:
    def __init__(self, corpus: Sequence[str] | None = None) -> None:
        samples = list(corpus or DEFAULT_TEXT_CORPUS)
        self.unigram: Counter[str] = Counter()
        self.bigram: Counter[Tuple[str, str]] = Counter()
        self.vocab: set[str] = set()
        for sent in samples:
            tokens = ["<s>"] + tokenize_text(sent) + ["</s>"]
            for tok in tokens:
                self.unigram[tok] += 1
                self.vocab.add(tok)
            for i in range(len(tokens) - 1):
                self.bigram[(tokens[i], tokens[i + 1])] += 1

    def score(self, tokens: Sequence[str]) -> float:
        seq = ["<s>"] + list(tokens) + ["</s>"]
        v = max(1, len(self.vocab))
        score = 0.0
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            numerator = self.bigram[(a, b)] + 1.0
            denominator = self.unigram[a] + float(v)
            score += math.log(numerator / denominator)
        return score

    def best_next(self, prev: str) -> str:
        candidates = [pair for pair in self.bigram.keys() if pair[0] == prev and pair[1] != "</s>"]
        if not candidates:
            return "you"
        best = max(candidates, key=lambda pair: self.bigram[pair])
        return best[1]


def label_to_words(label: str) -> List[str]:
    return tokenize_text(label.replace("_", " "))


def tokenize_text(text: str) -> List[str]:
    pieces = [p.strip().lower() for p in text.split()]
    return [p for p in pieces if p]


def detokenize(tokens: Sequence[str]) -> str:
    text = " ".join(tokens).strip()
    if not text:
        return ""
    if text.startswith("i "):
        text = "I " + text[2:]
    elif text == "i":
        text = "I"
    text = text[0].upper() + text[1:]
    if text.split()[0].lower() in {"how", "what", "where", "when", "why"}:
        if not text.endswith("?"):
            text = text.rstrip(".!?") + "?"
    else:
        if not text.endswith("."):
            text = text.rstrip(".!?") + "."
    return text


def compress_repetitions(tokens: Sequence[str]) -> List[str]:
    out: List[str] = []
    for tok in tokens:
        if not out or out[-1] != tok:
            out.append(tok)
    return out


class OnlineSentenceBuilder:
    def __init__(
        self,
        max_words: int = 10,
        cooldown_seconds: float = 1.1,
        lm: BigramLanguageModel | None = None,
    ) -> None:
        self.max_words = max(2, int(max_words))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._tokens: Deque[str] = deque(maxlen=self.max_words)
        self._last_emit_time = 0.0
        self._last_label = ""
        self.lm = lm or BigramLanguageModel()

    def add_label(self, label: str, ts: float | None = None) -> bool:
        now = float(ts if ts is not None else time.time())
        normalized = label.strip().lower()
        if not normalized:
            return False
        if normalized == self._last_label and (now - self._last_emit_time) < self.cooldown_seconds:
            return False

        words = label_to_words(normalized)
        if not words:
            return False
        changed = False
        for word in words:
            if not self._tokens or self._tokens[-1] != word:
                self._tokens.append(word)
                changed = True
        if changed:
            self._last_emit_time = now
            self._last_label = normalized
        return changed

    def clear(self) -> None:
        self._tokens.clear()
        self._last_emit_time = 0.0
        self._last_label = ""

    def words(self) -> List[str]:
        return list(self._tokens)

    def words_text(self) -> str:
        words = self.words()
        return " ".join(words) if words else ""

    def sentence(self) -> str:
        tokens = compress_repetitions(self.words())
        if not tokens:
            return ""

        # Lightweight NLP cleanup for common isolated ASL vocab.
        if tokens[0] in {"fine", "ok", "okay"}:
            tokens = ["i", "am"] + tokens
        if tokens[0] == "you" and len(tokens) == 1:
            tokens = ["are", "you"]

        # If sequence is very short, use a bigram LM to propose one extra connector token.
        if len(tokens) < 3:
            next_tok = self.lm.best_next(tokens[-1] if tokens else "<s>")
            if next_tok and next_tok not in {"</s>", "<s>"}:
                candidate = tokens + [next_tok]
                if self.lm.score(candidate) > self.lm.score(tokens):
                    tokens = candidate

        return detokenize(tokens)

