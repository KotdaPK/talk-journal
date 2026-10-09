import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from talk_journal.preprocessing.deepfilternet import DeepFilterNet3Preprocessor, _legacy_audio_compat
from talk_journal.preprocessing.none import NonePreprocessor


class PreprocessorTests(unittest.TestCase):
    def test_none_passes_mono_backend_rate_and_rejects_stereo(self):
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "source.wav", Path(temp) / "out.wav"
            sf.write(src, np.zeros(16000), 16000)
            NonePreprocessor().process(src, dst, None)
            self.assertEqual(src.read_bytes(), dst.read_bytes())
            sf.write(src, np.zeros((10, 2)), 48000)
            with self.assertRaisesRegex(ValueError, "mono"):
                NonePreprocessor().process(src, dst, None)

    def test_legacy_audio_shim_roundtrip(self):
        _legacy_audio_compat()
        from df.io import load_audio, save_audio
        from df.enhance import init_df
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "source.wav", Path(temp) / "out.wav"
            sf.write(src, np.zeros(480), 48000)
            audio, info = load_audio(src, sr=48000)
            self.assertEqual(info.sample_rate, 48000)
            self.assertEqual(tuple(audio.shape), (1, 480))
            save_audio(dst, audio, 48000)
            self.assertEqual(sf.info(dst).samplerate, 48000)
            self.assertTrue(callable(init_df))

    def test_deepfilternet_model_and_highpass(self):
        _legacy_audio_compat()
        with tempfile.TemporaryDirectory() as temp:
            src, dst = Path(temp) / "source.wav", Path(temp) / "out.wav"
            sf.write(src, np.zeros(480), 48000)
            state = MagicMock()
            state.sr.return_value = 48000
            model = MagicMock()
            with patch("df.enhance.init_df", return_value=(model, state, None)) as init, patch("df.enhance.enhance", side_effect=lambda _m, _s, audio, **kw: audio) as enhance:
                pre = DeepFilterNet3Preprocessor()
                pre.process(src, dst, 70)
                pre.close()
            init.assert_called_once_with(model_base_dir="DeepFilterNet3", log_file=None)
            self.assertEqual(enhance.call_args.kwargs, {"pad": True})
            self.assertEqual(sf.info(dst).samplerate, 48000)
            self.assertIsNone(pre._model)

if __name__ == "__main__": unittest.main()
