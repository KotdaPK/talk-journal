import sys
import types
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


class ParakeetBackendTests(unittest.TestCase):
    def _wav(self, directory: str, *, sample_rate: int = 16_000) -> Path:
        path = Path(directory) / "audio.wav"
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(b"\x00\x00" * sample_rate * 2)
        return path

    def test_loads_photon_once_on_cuda_and_closes_context(self):
        import talk_journal.backends.parakeet as parakeet

        speech = MagicMock()
        context = MagicMock()
        context.__enter__.return_value = speech
        photon = MagicMock(return_value=context)
        fake_moondream = types.SimpleNamespace(photon=photon)

        with patch.dict(sys.modules, {"moondream": fake_moondream}):
            backend = parakeet.ParakeetBackend()
            backend.load()
            backend.load()
            backend.close()
            backend.close()

        photon.assert_called_once_with("moondream/parakeet-ultra", device="cuda:0")
        context.__enter__.assert_called_once_with()
        context.__exit__.assert_called_once_with(None, None, None)

    def test_transcribes_16k_wav_with_word_timestamps_and_normalizes_result(self):
        import talk_journal.backends.parakeet as parakeet

        speech = MagicMock()
        speech.transcribe.return_value = types.SimpleNamespace(
            text="Hello world.",
            segments=[
                types.SimpleNamespace(
                    text="Hello world.",
                    start=0.0,
                    end=1.25,
                    words=[
                        types.SimpleNamespace(text="Hello", start=0.0, end=0.4),
                        types.SimpleNamespace(text="world.", start=0.5, end=1.25),
                    ],
                )
            ],
        )
        context = MagicMock()
        context.__enter__.return_value = speech
        fake_moondream = types.SimpleNamespace(photon=MagicMock(return_value=context))

        with TemporaryDirectory() as temporary, patch.dict(sys.modules, {"moondream": fake_moondream}):
            audio = self._wav(temporary)
            backend = parakeet.ParakeetBackend()
            result = backend.transcribe(audio, hotwords=[])

        speech.transcribe.assert_called_once_with(audio=audio, timestamps="word")
        self.assertEqual(result.text, "Hello world.")
        self.assertEqual(result.model_id, "moondream/parakeet-ultra")
        self.assertEqual(result.duration_seconds, 2.0)
        self.assertEqual(result.timestamp_precision, "word")
        self.assertEqual(result.segments[0].words[1].text, "world.")
        self.assertEqual(result.segments[0].end, 1.25)
        self.assertFalse(result.metadata["hotwords_supported"])

    def test_rejects_non_16k_wav_before_calling_photon(self):
        import talk_journal.backends.parakeet as parakeet

        with TemporaryDirectory() as temporary:
            audio = self._wav(temporary, sample_rate=8_000)
            backend = parakeet.ParakeetBackend()
            with self.assertRaisesRegex(ValueError, "16 kHz WAV"):
                backend.transcribe(audio, hotwords=[])


if __name__ == "__main__":
    unittest.main()
