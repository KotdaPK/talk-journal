"""Explicit, lazily imported backend and preprocessor registry."""
from __future__ import annotations

import importlib

BACKENDS = {
    "parakeet": ("talk_journal.backends.parakeet", "ParakeetBackend"),
    "moss": ("talk_journal.backends.moss", "MOSSBackend"),
    "granite": ("talk_journal.backends.granite", "GraniteBackend"),
}
PREPROCESSORS = {
    "none": ("talk_journal.preprocessing.none", "NonePreprocessor"),
    "deepfilternet": ("talk_journal.preprocessing.deepfilternet", "DeepFilterNet3Preprocessor"),
}


def make(registry: dict, name: str):
    if name not in registry:
        raise ValueError(f"Unknown selection {name!r}; choose {', '.join(registry)}")
    module, cls = registry[name]
    return getattr(importlib.import_module(module), cls)()
