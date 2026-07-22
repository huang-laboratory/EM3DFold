"""
Automated Protein, DNA and RNA structure modeling from cryo-EM maps
Tao Li et al.
"""

from pathlib import Path


def _resolve_runtime_log_dir(args):
    for attr_name in ("output", "output_dir", "out_dir"):
        value = getattr(args, attr_name, None)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.suffix.lower() in {".pdb", ".cif", ".mrc", ".map", ".json", ".npz", ".npy", ".txt", ".log"}:
            return path.parent
        return path

    output_pdb = getattr(args, "output_pdb", None)
    if output_pdb:
        return Path(output_pdb).expanduser().parent
    return None


def main():
    import time
    import platform
    import importlib
    import em3dfold
    from em3dfold.utils.log_utils import configure_runtime_logging

    # Check platform
    if platform.system() != "Linux":
        print("# WARN Your system is -> {}".format(platform.system()))
        print("# WARN This program only supports Linux systems (tested on CentOS 7)")
        print("# WARN This program will still run, but at any time it will crash")
        time.sleep(1)

    import argparse
    import warnings
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{getattr(em3dfold, '__version__')}",
    )

    # Suppress some warnings
    warnings.filterwarnings("ignore")

    modules = {
        "build": "em3dfold.build",
        "build_no_seq": "em3dfold.build_no_seq",
        "eval": "em3dfold.pipeline.eval",
        "pred": "em3dfold.pipeline.pred",
        "qscore": "em3dfold.pipeline.get_qscore",
        "hmm_search": "em3dfold.hmm_search",
        #"assemble": "em3dfold.pipeline.assemble",
    }

    subparsers = parser.add_subparsers(title="Choose a module",)
    subparsers.required = "True"

    for key, module_path in modules.items():
        module = None
        module_error = None
        try:
            module = importlib.import_module(module_path)
        except Exception as exc:
            module_error = exc

        module_parser = subparsers.add_parser(
            key,
            description=(module.__doc__ if module is not None else f"Module {module_path} is currently unavailable."),
            formatter_class=argparse.RawTextHelpFormatter,
        )
        module_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Also write full detailed logs to stdout; otherwise stdout shows only key stage messages.",
        )
        if module is not None:
            if hasattr(module, "add_args"):
                module.add_args(module_parser)
            elif hasattr(module, "build_arg_parser"):
                temp_parser = module.build_arg_parser()
                for action in temp_parser._actions:
                    if action.dest == "help":
                        continue
                    kwargs = {
                        "default": action.default,
                        "help": action.help,
                        "required": action.required,
                    }
                    if action.metavar is not None:
                        kwargs["metavar"] = action.metavar
                    if action.choices is not None:
                        kwargs["choices"] = action.choices
                    if action.nargs is not None:
                        kwargs["nargs"] = action.nargs
                    if action.const is not None:
                        kwargs["const"] = action.const
                    if getattr(action, "type", None) is not None:
                        kwargs["type"] = action.type
                    if action.option_strings:
                        if action.__class__.__name__ in {"_StoreTrueAction", "_StoreFalseAction"}:
                            kwargs["action"] = "store_true" if action.__class__.__name__ == "_StoreTrueAction" else "store_false"
                            kwargs.pop("type", None)
                            kwargs.pop("const", None)
                            kwargs.pop("nargs", None)
                        module_parser.add_argument(*action.option_strings, **kwargs)
                    else:
                        module_parser.add_argument(action.dest, **kwargs)
            else:
                raise AttributeError(
                    "Module {} must define add_args(parser) or build_arg_parser().".format(
                        module_path
                    )
                )
            module_parser.set_defaults(func=module.main, _module_key=key)
        else:
            def _raise_unavailable(_args, module_path=module_path, module_error=module_error):
                raise RuntimeError(
                    "Failed to import module {}: {}".format(module_path, module_error)
                )

            module_parser.set_defaults(func=_raise_unavailable, _module_key=key)

    args = parser.parse_args()
    runtime_log_dir = _resolve_runtime_log_dir(args)
    if runtime_log_dir is not None:
        module_key = getattr(args, "_module_key", "main")
        stdout_progress_logger_name = f"em3dfold.{module_key}.progress"
        if module_key in {"build", "build_no_seq"}:
            stdout_progress_logger_name = (
                "em3dfold.build.progress",
                "em3dfold.pred.progress",
                "em3dfold.infer.progress",
            )
            progress_logger_name = "em3dfold.build.progress"
        else:
            progress_logger_name = f"em3dfold.{module_key}.progress"
        configure_runtime_logging(
            runtime_log_dir,
            package_prefixes=("em3dfold", "em3dfit"),
            progress_logger_name=progress_logger_name,
            stdout_progress_logger_name=stdout_progress_logger_name,
            verbose=bool(getattr(args, "verbose", False)),
            excluded_prefixes=("em3dfold.rinalmo", "em3dfold.bin.src"),
            helper_modules=(
                "em3dfold.utils.log_utils",
                "em3dfold.utils.misc_utils",
                "em3dfit.log_utils",
                "em3dfit.utils",
            ),
        )
    args.func(args)

if __name__ == '__main__':
    main()

