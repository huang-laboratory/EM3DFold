import io
import sys
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from em3dfold import __main__


class CliLazyImportTests(unittest.TestCase):
    def test_only_selected_subcommand_is_imported(self):
        imported_modules = []

        def fake_import(module_path):
            imported_modules.append(module_path)
            if module_path != "em3dfold.pipeline.eval":
                raise AssertionError(f"unexpected import: {module_path}")
            return SimpleNamespace(
                __doc__="Evaluate a structure.",
                add_args=lambda parser: parser.add_argument("--target"),
                main=lambda args: None,
            )

        with patch.object(sys, "argv", ["em3dfold", "eval", "--help"]):
            with patch("importlib.import_module", side_effect=fake_import):
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        __main__.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(imported_modules, ["em3dfold.pipeline.eval"])

    def test_top_level_help_does_not_import_subcommands(self):
        with patch.object(sys, "argv", ["em3dfold", "--help"]):
            with patch("importlib.import_module") as import_module:
                with redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        __main__.main()

        self.assertEqual(raised.exception.code, 0)
        import_module.assert_not_called()


if __name__ == "__main__":
    unittest.main()
