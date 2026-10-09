"""DeepFilterNet3 preprocessing of decoded 48 kHz mono WAV."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any


def _legacy_audio_compat() -> None:
    """Adapt DF 0.5.6's df.io imports to torchaudio 2.11 without vendor edits.

    DF imports the now-removed torchaudio.backend.common.AudioMetaData.
    Only WAVs are sent to this adapter, so soundfile supplies its I/O.
    """
    if "df.io" in sys.modules:
        return
    import numpy as np
    import soundfile as sf
    import torch
    from torchaudio.functional import resample as torch_resample

    module = types.ModuleType("df.io")

    def load_audio(file, sr=None, **kwargs):
        data, native_rate = sf.read(str(file), dtype="float32", always_2d=True)
        tensor = torch.from_numpy(np.ascontiguousarray(data.T))
        if sr is not None and native_rate != sr:
            tensor = torch_resample(tensor, native_rate, sr)
        return tensor.contiguous(), types.SimpleNamespace(sample_rate=native_rate)

    def save_audio(file, audio, sr, **kwargs):
        samples = torch.as_tensor(audio).detach().cpu().numpy()
        sf.write(str(file), samples.T if samples.ndim == 2 else samples, sr, subtype="PCM_16")

    def resample(audio, orig_sr, new_sr, **kwargs):
        return torch_resample(audio, orig_sr, new_sr)

    module.load_audio = load_audio
    module.save_audio = save_audio
    module.resample = resample
    sys.modules["df.io"] = module


class DeepFilterNet3Preprocessor:
    name = "deepfilternet"

    def __init__(self) -> None:
        self._model: Any | None = None
        self._df_state: Any | None = None

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            _legacy_audio_compat()
            from df.enhance import init_df
        except ImportError as exc:
            raise RuntimeError("DeepFilterNet3 dependency missing; rerun install.ps1 or install.sh") from exc
        self._model, self._df_state, _ = init_df(model_base_dir="DeepFilterNet3", log_file=None)
        if self._df_state.sr() != 48_000:
            self.close()
            raise RuntimeError("DeepFilterNet3 requires a 48 kHz model state")

    def process(self, source: Path, destination: Path, highpass_hz: int | None) -> None:
        import soundfile as sf
        import torch
        info = sf.info(source)
        if info.samplerate != 48_000 or info.channels != 1:
            raise ValueError("DeepFilterNet3 needs decoded 48 kHz mono WAV")
        self.load()
        from df.enhance import enhance
        from df.io import load_audio, save_audio
        audio, _ = load_audio(source, sr=48_000)
        if highpass_hz is not None:
            from scipy.signal import butter, sosfilt
            sos = butter(4, highpass_hz, btype="highpass", fs=48_000, output="sos")
            audio = torch.as_tensor(sosfilt(sos, audio.cpu().numpy(), axis=-1).copy(), dtype=audio.dtype)
        with torch.inference_mode():
            enhanced = enhance(self._model, self._df_state, audio, pad=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_audio(destination, enhanced, sr=48_000)

    def close(self) -> None:
        self._model = self._df_state = None
