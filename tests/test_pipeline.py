import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from talk_journal import output, pipeline
from talk_journal.models import TranscriptResult


class Backend:
    model_id = "mock/model"
    sample_rate = 16000
    supports_hotwords = True
    revision = "r1"
    def __init__(self):
        self.loads = 0
        self.calls = 0
    def load(self):
        self.loads += 1
    def transcribe(self, audio, hotwords):
        self.calls += 1
        return TranscriptResult("hello", self.model_id, self.revision, 1.0)
    def close(self): pass


class Enhancer:
    def load(self): pass
    def process(self, src, dst, highpass_hz): dst.write_bytes(src.read_bytes())
    def close(self): pass


class PipelineTests(unittest.TestCase):
    def test_idempotence_config_changes_collision_and_source_modification(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.wav").write_bytes(b"one")
            (folder / "a.m4a").write_bytes(b"two")
            (folder / "sub").mkdir()
            (folder / "sub" / "a.wav").write_bytes(b"three")
            backend, enhancer = Backend(), Enhancer()
            def maker(registry, name):
                return backend if registry is pipeline.BACKENDS else enhancer
            def decode(src, dst, rate): dst.write_bytes(src.read_bytes())
            with patch.object(pipeline, "make", side_effect=maker), patch.object(pipeline.audio, "decode", side_effect=decode):
                kwargs = dict(output_dir=folder / "out", model="parakeet", preprocess="none", recursive=True)
                self.assertEqual(pipeline.process(folder, **kwargs), (3, 0, 0))
                self.assertEqual(pipeline.process(folder, **kwargs), (0, 0, 3))
                self.assertEqual(backend.loads, 1)
                self.assertEqual(pipeline.process(folder, **{**kwargs, "model": "moss"}), (3, 0, 0))
                self.assertEqual(pipeline.process(folder, **{**kwargs, "preprocess": "deepfilternet"}), (3, 0, 0))
                self.assertEqual(pipeline.process(folder, **{**kwargs, "model": "moss", "preprocess": "deepfilternet"}), (3, 0, 0))
                (folder / "a.wav").write_bytes(b"changed")
                self.assertEqual(pipeline.process(folder, **kwargs), (1, 0, 2))
                self.assertEqual(pipeline.process(folder, **{**kwargs, "overwrite": True}), (3, 0, 0))
            self.assertEqual(len(list((folder / "out").rglob("metadata.json"))), 12)

    def test_failed_write_has_no_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            config = {"model": "parakeet", "preprocess": "none"}
            source = {"sha256": "test"}
            result = TranscriptResult("hello", "mock/model", None, 1.0)
            with patch.object(output, "atomic_text", side_effect=OSError("interrupted")):
                with self.assertRaises(OSError): output.save(directory, source, config, result)
            self.assertFalse(output.manifest_matches(directory, source, config))
            output.save(directory, source, config, result)
            self.assertTrue(output.manifest_matches(directory, source, config))
            (directory / "transcript.txt").write_text("corrupted")
            self.assertFalse(output.manifest_matches(directory, source, config))
            output.save(directory, source, config, TranscriptResult("", "mock/model", None, 1.0))
            self.assertFalse(output.manifest_matches(directory, source, config))

    def test_selected_overwrite_invalidates_previous_marker_before_model_load(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.wav").write_bytes(b"a")
            backend, enhancer = Backend(), Enhancer()
            def maker(registry, name): return backend if registry is pipeline.BACKENDS else enhancer
            with patch.object(pipeline, "make", side_effect=maker), patch.object(
                pipeline.audio, "decode", side_effect=lambda src, dst, rate: dst.write_bytes(src.read_bytes())
            ):
                self.assertEqual(pipeline.process(folder, folder / "out", preprocess="none"), (1, 0, 0))
                with patch.object(backend, "load", side_effect=RuntimeError("model loading failed")):
                    with self.assertRaisesRegex(RuntimeError, "model loading failed"):
                        pipeline.process(folder, folder / "out", preprocess="none", overwrite=True)
            self.assertEqual(len(list((folder / "out").rglob("metadata.json"))), 0)

    def test_hotword_capability_checked_before_load(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.wav").write_bytes(b"a")
            backend = Backend()
            backend.supports_hotwords = False
            with patch.object(pipeline, "make", return_value=backend):
                with self.assertRaisesRegex(ValueError, "does not support"):
                    pipeline.process(folder, keywords=["keyword"])
            self.assertEqual(backend.loads, 0)

    def test_failed_file_does_not_stop_next(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.wav").write_bytes(b"a")
            (folder / "b.wav").write_bytes(b"b")
            backend = Backend()
            enhancer = Enhancer()
            def maker(registry, name): return backend if registry is pipeline.BACKENDS else enhancer
            def decode(src, dst, rate):
                if src.name == "a.wav": raise ValueError("bad")
                dst.write_bytes(src.read_bytes())
            with patch.object(pipeline, "make", side_effect=maker), patch.object(pipeline.audio, "decode", side_effect=decode):
                self.assertEqual(pipeline.process(folder, folder / "out", preprocess="none"), (1, 1, 0))
            self.assertEqual(len(list((folder / "out").rglob("metadata.json"))), 1)

    def test_empty_transcript_does_not_mark_recording_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.wav").write_bytes(b"speech fixture")
            backend, enhancer = Backend(), Enhancer()
            backend.transcribe = lambda audio, hotwords: TranscriptResult("", backend.model_id, backend.revision, 1.0)
            def maker(registry, name): return backend if registry is pipeline.BACKENDS else enhancer
            with patch.object(pipeline, "make", side_effect=maker), patch.object(
                pipeline.audio, "decode", side_effect=lambda src, dst, rate: dst.write_bytes(src.read_bytes())
            ):
                self.assertEqual(pipeline.process(folder, folder / "out", preprocess="none"), (0, 1, 0))
            self.assertEqual(list((folder / "out").rglob("metadata.json")), [])

if __name__ == "__main__": unittest.main()
