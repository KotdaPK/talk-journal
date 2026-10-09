import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import transcribe_folder as cli


class CliTests(unittest.TestCase):
    def test_single_file_default_output_beside_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.m4a"
            source.touch()
            with patch("sys.argv", ["talk-journal", str(source)]), patch.object(
                cli, "load_models", return_value=(object(), MagicMock(), object(), "cpu")
            ), patch.object(cli, "transcribe_file", return_value="hello"):
                cli.main()
            self.assertEqual((Path(tmp) / "a.txt").read_text(), "hello")

    def test_folder_default_output_and_idempotent_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "audio"
            folder.mkdir()
            (folder / "a.m4a").touch()
            with patch("sys.argv", ["talk-journal", str(folder)]), patch.object(
                cli, "load_models", return_value=(object(), MagicMock(), object(), "cpu")
            ) as models, patch.object(cli, "transcribe_file", return_value="hello"):
                cli.main()
                cli.main()
            self.assertEqual((folder / "transcripts" / "a.txt").read_text(), "hello")
            self.assertEqual(models.call_count, 1)

    def test_collision_fails_before_model_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "a.mp3").touch()
            (folder / "a.m4a").touch()
            with patch("sys.argv", ["talk-journal", str(folder)]), patch.object(cli, "load_models") as models:
                with self.assertRaises(SystemExit):
                    cli.main()
            models.assert_not_called()
            self.assertFalse((folder / "transcripts" / "a.txt").exists())

    def test_partial_failures_return_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "a.m4a"
            source.touch()
            with patch("sys.argv", ["talk-journal", str(source)]), patch.object(
                cli, "load_models", return_value=(object(), MagicMock(), object(), "cpu")
            ), patch.object(cli, "transcribe_file", side_effect=RuntimeError("bad audio")):
                with self.assertRaises(SystemExit) as result:
                    with contextlib.redirect_stdout(io.StringIO()):
                        cli.main()
            self.assertEqual(result.exception.code, 1)
            self.assertFalse((Path(tmp) / "a.txt").exists())

    def test_bundled_ffmpeg_when_system_missing(self):
        with patch.object(cli.shutil, "which", return_value=None), patch.object(
            cli.imageio_ffmpeg, "get_ffmpeg_exe", return_value="/bundled/ffmpeg"
        ):
            self.assertEqual(cli.ffmpeg_executable(), "/bundled/ffmpeg")


if __name__ == "__main__":
    unittest.main()
