import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from talk_journal.backends.moss import MOSSBackend
from talk_journal.models import TranscriptResult


class MossBackendTests(unittest.TestCase):
    def _install_runtime(self, *, cuda=True, generate_side_effect=None):
        self.torch = types.SimpleNamespace(
            bfloat16="bf16",
            cuda=types.SimpleNamespace(is_available=MagicMock(return_value=cuda), empty_cache=MagicMock()),
        )
        self.model = MagicMock()
        self.processor = MagicMock()
        self.auto_model = MagicMock()
        self.auto_model.from_pretrained.return_value = self.model
        self.auto_processor = MagicMock()
        self.auto_processor.from_pretrained.return_value = self.processor
        self.generate = MagicMock(return_value={
            "text": "[0.00][S01]hello[1.00]",
            "generated_tokens": 4,
        })
        if generate_side_effect is not None:
            self.generate.side_effect = generate_side_effect
        self.build_messages = MagicMock(return_value=[{"role": "user"}])
        self.parse = MagicMock(return_value=[
            types.SimpleNamespace(start=0.0, end=1.0, speaker="S01", text=" hello ")
        ])
        transformers = types.ModuleType("transformers")
        transformers.AutoModelForCausalLM = self.auto_model
        transformers.AutoProcessor = self.auto_processor
        inference = types.ModuleType("moss_transcribe_diarize.inference_utils")
        inference.build_transcription_messages = self.build_messages
        inference.generate_transcription = self.generate
        self.device = types.SimpleNamespace(type="cuda")
        inference.resolve_device = MagicMock(return_value=self.device)
        moss = types.ModuleType("moss_transcribe_diarize")
        moss.parse_transcript = self.parse
        return patch.dict(sys.modules, {
            "torch": self.torch,
            "transformers": transformers,
            "moss_transcribe_diarize": moss,
            "moss_transcribe_diarize.inference_utils": inference,
        })

    def test_load_requires_cuda_without_importing_model_runtime(self):
        with self._install_runtime(cuda=False):
            backend = MOSSBackend()
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                backend.load()
        self.auto_model.from_pretrained.assert_not_called()

    def test_loads_native_model_and_processor_on_cuda(self):
        with self._install_runtime():
            backend = MOSSBackend()
            backend.load()
        self.auto_model.from_pretrained.assert_called_once_with(
            "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            trust_remote_code=True,
            revision=backend.revision,
            dtype="auto",
            attn_implementation="sdpa",
        )
        self.model.to.assert_any_call(dtype="bf16")
        self.model.eval.assert_called_once_with()
        self.auto_processor.from_pretrained.assert_called_once_with(
            "OpenMOSS-Team/MOSS-Transcribe-Diarize", trust_remote_code=True, revision=backend.revision
        )

    def test_transcribe_uses_official_helpers_hotword_prompt_and_normalizes_segments(self):
        with self._install_runtime(), tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "clip.wav"
            import soundfile as sf
            import numpy as np
            sf.write(audio, np.zeros(16000), 16000)
            backend = MOSSBackend()
            backend.load()
            result = backend.transcribe(audio, ["OpenMOSS", "Kotda"])
        self.build_messages.assert_called_once()
        prompt = self.build_messages.call_args.kwargs["prompt"]
        self.assertIn("Hotwords: OpenMOSS, Kotda", prompt)
        self.generate.assert_called_once_with(
            self.model,
            self.processor,
            self.build_messages.return_value,
            max_new_tokens=8192,
            do_sample=False,
            device=self.device,
            dtype="bf16",
        )
        self.parse.assert_called_once_with("[0.00][S01]hello[1.00]")
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.timestamp_precision, "segment")
        self.assertEqual(result.segments[0].speaker, "S01")
        self.assertEqual(result.segments[0].text, "hello")

    def test_output_at_generation_limit_uses_overlap_chunk_fallback(self):
        outputs = [
            {"text": "[0.00][S01]partial", "generated_tokens": 8192},
            {"text": "[0.00][S01]first[2.00]", "generated_tokens": 3},
            {"text": "[0.00][S01]second[2.00]", "generated_tokens": 3},
        ]
        with self._install_runtime(generate_side_effect=outputs), tempfile.TemporaryDirectory() as tmp, patch.object(
            MOSSBackend, "_decode_chunks", return_value=[
                (Path(tmp) / "one.wav", 0.0, 2.0),
                (Path(tmp) / "two.wav", 1.5, 3.5),
            ]
        ):
            audio = Path(tmp) / "long.wav"
            audio.touch()
            backend = MOSSBackend(chunk_seconds=2, overlap_seconds=0.5)
            backend.load()
            result = backend.transcribe(audio, [])
        self.assertEqual(self.generate.call_count, 3)
        self.assertEqual([segment.speaker for segment in result.segments], ["chunk-000:S01", "chunk-001:S01"])
        self.assertEqual([(segment.start, segment.end) for segment in result.segments], [(0.0, 1.0), (1.5, 2.5)])
        self.assertEqual(result.metadata["fallback"], "overlap-chunks")

    def test_cuda_oom_uses_overlap_chunk_fallback(self):
        outputs = [
            RuntimeError("CUDA out of memory"),
            {"text": "[0.00][S01]recovered[1.00]", "generated_tokens": 3},
        ]
        with self._install_runtime(generate_side_effect=outputs), tempfile.TemporaryDirectory() as tmp, patch.object(
            MOSSBackend, "_decode_chunks", return_value=[(Path(tmp) / "one.wav", 0.0, 1.0)]
        ):
            audio = Path(tmp) / "clip.wav"
            audio.touch()
            backend = MOSSBackend()
            backend.load()
            result = backend.transcribe(audio, [])
        self.assertEqual(result.metadata["fallback_cause"], "cuda-oom")
        self.assertEqual(result.segments[0].speaker, "chunk-000:S01")

    def test_non_oom_native_failure_is_not_hidden_by_chunk_fallback(self):
        with self._install_runtime(generate_side_effect=RuntimeError("bad model state")), tempfile.TemporaryDirectory() as tmp, patch.object(
            MOSSBackend, "_decode_chunks"
        ) as chunks:
            audio = Path(tmp) / "clip.wav"
            audio.touch()
            backend = MOSSBackend()
            backend.load()
            with self.assertRaisesRegex(RuntimeError, "bad model state"):
                backend.transcribe(audio, [])
        chunks.assert_not_called()

    def test_real_chunk_decoder_covers_source_with_overlap(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            sf.write(source, np.zeros(5 * 16000), 16000)
            backend = MOSSBackend(chunk_seconds=2, overlap_seconds=0.5)
            chunks = backend._decode_chunks(source, Path(temp))
            self.assertEqual([(start, end) for _, start, end in chunks],
                             [(0.0, 2.0), (1.5, 3.5), (3.0, 5.0)])
            self.assertEqual(sum(sf.info(path).duration for path, _, _ in chunks), 6.0)

    def test_adaptive_chunk_size_retries_oom_and_truncated_output(self):
        with self._install_runtime():
            backend = MOSSBackend(chunk_seconds=600)
            backend.load()
            answer = TranscriptResult("ok", backend.model_id, backend.revision, 1.0)
            with patch.object(backend, "_transcribe_chunks", side_effect=[
                RuntimeError("MOSS ran out of CUDA memory even after overlap chunk fallback."),
                RuntimeError("MOSS chunk output reached max_new_tokens; refusing a silently truncated transcript."),
                answer,
            ]) as chunk:
                result = backend._adaptive_chunks(Path("dummy.wav"), "prompt", cause="cuda-oom")
            self.assertIs(result, answer)
            self.assertEqual(chunk.call_count, 3)
            self.assertEqual(backend.chunk_seconds, 150)


if __name__ == "__main__":
    unittest.main()
