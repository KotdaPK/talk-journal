import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from talk_journal import cli
from talk_journal.config import BACKENDS, PREPROCESSORS
from talk_journal.backends.granite import GraniteBackend, build_prompt


class CliTests(unittest.TestCase):
    def test_registry_and_defaults(self):
        self.assertEqual(set(BACKENDS), {"parakeet", "moss", "granite"})
        self.assertEqual(set(PREPROCESSORS), {"deepfilternet", "none"})
        with tempfile.TemporaryDirectory() as temp, patch.object(cli, "process", return_value=(0, 0, 0)) as proc:
            cli.main([temp])
            self.assertEqual(proc.call_args.kwargs["model"], "parakeet")
            self.assertEqual(proc.call_args.kwargs["preprocess"], "deepfilternet")

    def test_selection_forwarding_and_exit(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cli, "process", return_value=(0, 1, 0)) as proc:
            self.assertEqual(cli.main([temp, "--model", "granite", "--preprocess", "none", "--recursive", "--overwrite", "--keywords", "name"]), 1)
            self.assertEqual(proc.call_args.kwargs["model"], "granite")
            self.assertEqual(proc.call_args.kwargs["keywords"], ["name"])
            self.assertTrue(proc.call_args.kwargs["recursive"])

    def test_granite_cuda_fail_closed(self):
        import torch
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                GraniteBackend().load()

    def test_granite_keyword_prompt(self):
        from unittest.mock import Mock
        tokenizer = Mock()
        build_prompt(tokenizer, ["Kotda"])
        self.assertIn("Keywords: Kotda", tokenizer.apply_chat_template.call_args.args[0][0]["content"])

    def test_list_commands_need_no_input_or_models(self):
        self.assertEqual(cli.main(["--list-models"]), 0)
        self.assertEqual(cli.main(["--list-preprocessors"]), 0)

if __name__ == "__main__": unittest.main()
