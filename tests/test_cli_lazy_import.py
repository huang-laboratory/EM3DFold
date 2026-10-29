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

    def test_cpu_limit_is_applied_to_build(self):
        module = SimpleNamespace(
            __doc__="Build a structure.",
            add_args=lambda parser: None,
            main=lambda args: None,
        )
        cpu_config = SimpleNamespace(
            max_threads=4,
            affinity_before=tuple(range(16)),
            affinity_after=(0, 1, 2, 3),
        )
        with patch.object(sys, "argv", ["em3dfold", "build"]):
            with patch("importlib.import_module", return_value=module):
                with patch.object(
                    __main__,
                    "configure_cpu_resources",
                    return_value=cpu_config,
                ) as configure_cpu:
                    with patch.object(
                        __main__,
                        "configure_loaded_thread_pools",
                    ) as configure_pools:
                        __main__.main()

        configure_cpu.assert_called_once_with()
        configure_pools.assert_called_once_with(4)

    def test_cpu_limit_is_not_applied_to_standalone_hmm_search(self):
        module = SimpleNamespace(
            __doc__="Search HMM profiles.",
            add_args=lambda parser: None,
            main=lambda args: None,
        )
        with patch.object(sys, "argv", ["em3dfold", "hmm_search"]):
            with patch("importlib.import_module", return_value=module):
                with patch.object(
                    __main__,
                    "configure_cpu_resources",
                ) as configure_cpu:
                    with patch.object(
                        __main__,
                        "configure_loaded_thread_pools",
                    ) as configure_pools:
                        __main__.main()

        configure_cpu.assert_not_called()
        configure_pools.assert_not_called()


if __name__ == "__main__":
    unittest.main()
