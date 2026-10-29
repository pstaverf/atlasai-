"""Command-line tools for configuration, diagnostics, tokenizer, dataset, model, and LLM."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from atlasai import __version__
from atlasai.config import DEFAULT_CONFIG_PATH, AtlasConfig, ConfigError, load_config
from atlasai.core.model.base import LLMError
from atlasai.core.tokenizer.bpe import BPETokenizer, TokenizerError, train_bpe
from atlasai.training.dataset import DatasetError, load_dataset, read_documents
from atlasai.training.errors import TrainingError

_REQUIRED_PYTHON = (3, 12)


def _check(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _linux_available_memory_mib() -> int | None:
    """Read host MemAvailable; this is not a container/cgroup memory limit."""
    try:
        lines = Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    for line in lines:
        if line.startswith("MemAvailable:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    return int(parts[1]) // 1024
                except ValueError:
                    return None
    return None


def doctor_report(config: AtlasConfig, config_path: Path) -> dict[str, object]:
    """Build a machine-readable report without importing torch or loading weights."""
    checks = [_check("config", "ok", "Configuration loaded and validated")]
    python_ok = sys.version_info[:2] == _REQUIRED_PYTHON
    checks.append(
        _check(
            "python",
            "ok" if python_ok else "error",
            f"Running {platform.python_version()}; required 3.12.x",
        )
    )

    try:
        torch_version: str | None = version("torch")
    except PackageNotFoundError:
        torch_version = None
    checks.append(
        _check(
            "torch",
            "ok" if torch_version is not None else "info",
            f"Installed version {torch_version}"
            if torch_version is not None
            else "Optional until the model stages",
        )
    )

    artifacts: dict[str, object] = {}
    for name, path in (
        ("tokenizer", config.model.tokenizer_path),
        ("checkpoint", config.model.checkpoint_path),
    ):
        exists = path.is_file()
        artifacts[name] = {"path": str(path), "exists": exists}
        checks.append(
            _check(
                name,
                "ok" if exists else "info",
                "File exists; doctor does not load or validate its contents"
                if exists
                else "Not created yet",
            )
        )

    cpu_count = os.cpu_count()
    memory_mib = _linux_available_memory_mib()
    if cpu_count is not None and config.runtime.num_threads > cpu_count:
        checks.append(
            _check("threads", "warning", "Configured threads exceed visible logical CPUs")
        )
    if memory_mib is not None and memory_mib < 1536:
        checks.append(
            _check(
                "memory",
                "warning",
                "Less than 1.5 GiB available; 10M-parameter training may be constrained",
            )
        )

    statuses = {check["status"] for check in checks}
    status = "error" if "error" in statuses else "warning" if "warning" in statuses else "ok"
    return {
        "stage": 4,
        "status": status,
        "config_path": str(config_path),
        "backend": config.model.backend,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "resources": {
            "logical_cpus": cpu_count,
            "host_available_memory_mib": memory_mib,
            "configured_threads": config.runtime.num_threads,
        },
        "torch": {"installed": torch_version is not None, "version": torch_version},
        "artifacts": artifacts,
        "model_loaded": False,
        "model_status": "local_backend_implemented",
        "checks": checks,
    }


def _common_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML config path (default: configs/atlas.toml, relative to the current directory)",
    )
    return common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlasai", description="Atlas AI local-first toolkit")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = _common_parser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", parents=[common], help="Print the validated config as JSON")
    commands.add_parser("doctor", parents=[common], help="Inspect the environment as JSON")

    tokenizer = commands.add_parser("tokenizer", help="Train and use the byte-level BPE tokenizer")
    tokenizer_commands = tokenizer.add_subparsers(dest="tokenizer_command", required=True)
    train = tokenizer_commands.add_parser(
        "train", parents=[common], help="Train BPE on UTF-8 files"
    )
    train.add_argument("--input", nargs="*", type=Path, default=None, help="Corpus .txt files")
    train.add_argument("--output", type=Path, default=None, help="Tokenizer JSON output path")
    train.add_argument("--vocab-size", type=int, default=None)
    train.add_argument("--min-frequency", type=int, default=None)

    encode = tokenizer_commands.add_parser("encode", parents=[common], help="Encode text to ids")
    encode.add_argument("--tokenizer", type=Path, default=None)
    encode.add_argument("--text", required=True)
    encode.add_argument("--bos", action="store_true")
    encode.add_argument("--eos", action="store_true")

    decode = tokenizer_commands.add_parser("decode", parents=[common], help="Decode ids to text")
    decode.add_argument("--tokenizer", type=Path, default=None)
    decode.add_argument("--ids", required=True, help="Comma-separated token ids")
    decode.add_argument("--keep-special", action="store_true")

    dataset = commands.add_parser("dataset", help="Prepare next-token training blocks")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = dataset_commands.add_parser("inspect", parents=[common], help="Print dataset stats")
    inspect.add_argument("--input", nargs="*", type=Path, default=None)
    inspect.add_argument("--tokenizer", type=Path, default=None)
    inspect.add_argument("--context-length", type=int, default=None)
    inspect.add_argument("--stride", type=int, default=None)
    inspect.add_argument("--val-fraction", type=float, default=None)
    inspect.add_argument("--seed", type=int, default=None)

    model = commands.add_parser("model", help="Train or evaluate the own CPU Transformer")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    train_model_parser = model_commands.add_parser(
        "train", parents=[common], help="Train and save the best validation checkpoint"
    )
    evaluate_parser = model_commands.add_parser(
        "evaluate", parents=[common], help="Safely reload and evaluate a checkpoint"
    )
    for command in (train_model_parser, evaluate_parser):
        command.add_argument("--input", nargs="*", type=Path, default=None)
        command.add_argument("--tokenizer", type=Path, default=None)
        command.add_argument("--checkpoint", type=Path, default=None)
        command.add_argument("--val-fraction", type=float, default=None)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--batch-size", type=int, default=None)
    train_model_parser.add_argument("--context-length", type=int, default=None)
    train_model_parser.add_argument("--stride", type=int, default=None)
    train_model_parser.add_argument("--max-steps", type=int, default=None)
    train_model_parser.add_argument("--quiet", action="store_true")

    llm = commands.add_parser("llm", help="Use the own local LLM backend")
    llm_commands = llm.add_subparsers(dest="llm_command", required=True)
    generate = llm_commands.add_parser("generate", parents=[common], help="Generate text locally")
    generate.add_argument("--prompt", required=True)
    generate.add_argument("--max-new-tokens", type=int, default=None)
    generate.add_argument("--temperature", type=float, default=None)
    generate.add_argument("--seed", type=int, default=None)
    info = llm_commands.add_parser(
        "info", parents=[common], help="Inspect the loaded local backend"
    )
    for command in (generate, info):
        command.add_argument("--tokenizer", type=Path, default=None)
        command.add_argument("--checkpoint", type=Path, default=None)
    return parser


def _input_paths(args: argparse.Namespace, config: AtlasConfig) -> list[Path]:
    values = args.input if args.input is not None else list(config.data.input_paths)
    if not values:
        raise DatasetError("No input documents: pass --input or set data.input_paths")
    return [Path(value).expanduser().resolve(strict=False) for value in values]


def _tokenizer_path(args: argparse.Namespace, config: AtlasConfig) -> Path:
    value = args.tokenizer if args.tokenizer is not None else config.model.tokenizer_path
    return Path(value).expanduser().resolve(strict=False)


def _run_tokenizer(args: argparse.Namespace, config: AtlasConfig) -> dict[str, object]:
    if args.tokenizer_command == "train":
        paths = _input_paths(args, config)
        documents = list(read_documents(paths))
        vocab_size = args.vocab_size if args.vocab_size is not None else config.tokenizer.vocab_size
        min_frequency = (
            args.min_frequency if args.min_frequency is not None else config.tokenizer.min_frequency
        )
        tokenizer = train_bpe(documents, vocab_size=vocab_size, min_frequency=min_frequency)
        output = (
            Path(args.output).expanduser().resolve(strict=False)
            if args.output is not None
            else config.model.tokenizer_path
        )
        tokenizer.save(output)
        return {
            "command": "tokenizer.train",
            "output": str(output),
            "documents": len(documents),
            "vocab_size": tokenizer.vocab_size,
            "merges": len(tokenizer.merges),
        }
    if args.tokenizer_command == "encode":
        tokenizer = BPETokenizer.load(_tokenizer_path(args, config))
        ids = tokenizer.encode(args.text, add_bos=args.bos, add_eos=args.eos)
        return {"command": "tokenizer.encode", "ids": ids, "count": len(ids)}
    if args.tokenizer_command == "decode":
        tokenizer = BPETokenizer.load(_tokenizer_path(args, config))
        try:
            ids = [int(part.strip()) for part in args.ids.split(",") if part.strip()]
        except ValueError as exc:
            raise TokenizerError(f"Invalid token id list: {args.ids!r}") from exc
        text = tokenizer.decode(ids, skip_special_tokens=not args.keep_special)
        return {"command": "tokenizer.decode", "text": text}
    raise ConfigError(f"Unknown tokenizer command: {args.tokenizer_command}")


def _run_dataset(args: argparse.Namespace, config: AtlasConfig) -> dict[str, object]:
    if args.dataset_command != "inspect":
        raise ConfigError(f"Unknown dataset command: {args.dataset_command}")
    paths = _input_paths(args, config)
    tokenizer = BPETokenizer.load(_tokenizer_path(args, config))
    context_length = (
        args.context_length if args.context_length is not None else config.model.context_length
    )
    stride = args.stride if args.stride is not None else config.data.stride
    val_fraction = args.val_fraction if args.val_fraction is not None else config.data.val_fraction
    seed = args.seed if args.seed is not None else config.runtime.seed
    dataset = load_dataset(
        paths,
        tokenizer,
        context_length=context_length,
        stride=stride,
        val_fraction=val_fraction,
        seed=seed,
    )
    payload: dict[str, object] = dict(dataset.stats.to_dict())
    payload["command"] = "dataset.inspect"
    payload["first_train_block"] = (
        {
            "input_ids": list(dataset.train[0].input_ids),
            "target_ids": list(dataset.train[0].target_ids),
        }
        if dataset.train
        else None
    )
    return payload


def _run_model(args: argparse.Namespace, config: AtlasConfig) -> dict[str, object]:
    # Keep config/doctor/tokenizer/dataset commands usable without PyTorch.
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is required for model commands. "
            "Install the CPU build using README instructions"
        ) from exc
    from atlasai.core.model.transformer import AtlasTransformer
    from atlasai.training.checkpoint import load_model_checkpoint
    from atlasai.training.evaluation import evaluate
    from atlasai.training.trainer import configure_torch, train_model

    paths = _input_paths(args, config)
    tokenizer_path = _tokenizer_path(args, config)
    tokenizer = BPETokenizer.load(tokenizer_path)
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve(strict=False)
        if args.checkpoint is not None
        else config.model.checkpoint_path
    )
    val_fraction = args.val_fraction if args.val_fraction is not None else config.data.val_fraction
    seed = args.seed if args.seed is not None else config.runtime.seed
    batch_size = args.batch_size if args.batch_size is not None else config.training.batch_size
    if args.model_command == "train":
        forbidden = {tokenizer_path, *paths}
        if checkpoint_path in forbidden or checkpoint_path.with_suffix(".metrics.json") in paths:
            raise TrainingError("Model outputs must not overwrite the tokenizer or corpus")
        context_length = (
            args.context_length if args.context_length is not None else config.model.context_length
        )
        stride = args.stride if args.stride is not None else config.data.stride
        dataset = load_dataset(
            paths,
            tokenizer,
            context_length=context_length,
            stride=stride,
            val_fraction=val_fraction,
            seed=seed,
        )
        training_config = replace(
            config.training,
            max_steps=args.max_steps if args.max_steps is not None else config.training.max_steps,
            batch_size=batch_size,
        )
        configure_torch(seed=seed, num_threads=config.runtime.num_threads)
        model = AtlasTransformer(
            vocab_size=tokenizer.vocab_size,
            context_length=context_length,
            config=config.transformer,
        )
        result = train_model(
            model,
            dataset,
            tokenizer_path=tokenizer_path,
            checkpoint_path=checkpoint_path,
            config=training_config,
            seed=seed,
            num_threads=config.runtime.num_threads,
            verbose=not args.quiet,
        )
        payload: dict[str, object] = dict(result.to_dict())
        payload["command"] = "model.train"
        payload["dataset"] = dataset.stats.to_dict()
        return payload
    if args.model_command == "evaluate":
        model, info = load_model_checkpoint(checkpoint_path, tokenizer_path=tokenizer_path)
        configure_torch(seed=seed, num_threads=config.runtime.num_threads)
        dataset = load_dataset(
            paths,
            tokenizer,
            context_length=model.context_length,
            stride=model.context_length,
            val_fraction=val_fraction,
            seed=seed,
        )
        metrics = evaluate(model, dataset.val, batch_size=batch_size, device=torch.device("cpu"))
        return {
            "command": "model.evaluate",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint": info.to_dict(),
            "validation": metrics.to_dict(),
            "dataset": dataset.stats.to_dict(),
        }
    raise ConfigError(f"Unknown model command: {args.model_command}")


def _run_llm(args: argparse.Namespace, config: AtlasConfig) -> dict[str, object]:
    try:
        from atlasai.core.model.local import AtlasLocalModel
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "PyTorch is required for the local LLM backend. "
            "Install the CPU build using README instructions"
        ) from exc
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve(strict=False)
        if args.checkpoint is not None
        else config.model.checkpoint_path
    )
    seed = args.seed if getattr(args, "seed", None) is not None else config.runtime.seed
    backend = AtlasLocalModel(
        checkpoint_path,
        _tokenizer_path(args, config),
        max_new_tokens=config.model.max_new_tokens,
        temperature=config.model.temperature,
        num_threads=config.runtime.num_threads,
        seed=seed,
    )
    if args.llm_command == "info":
        payload: dict[str, object] = dict(backend.info())
        payload["command"] = "llm.info"
        return payload
    if args.llm_command == "generate":
        result = backend.generate(
            args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        payload = dict(result.to_dict())
        payload["command"] = "llm.generate"
        payload["parameter_count"] = backend.parameter_count
        return payload
    raise ConfigError(f"Unknown llm command: {args.llm_command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve(strict=False)
        config = load_config(config_path)
        report: dict[str, object]
        if args.command == "config":
            report = config.to_dict()
        elif args.command == "doctor":
            report = doctor_report(config, config_path)
        elif args.command == "tokenizer":
            report = _run_tokenizer(args, config)
        elif args.command == "dataset":
            report = _run_dataset(args, config)
        elif args.command == "model":
            report = _run_model(args, config)
        elif args.command == "llm":
            report = _run_llm(args, config)
        else:
            raise ConfigError(f"Unknown command: {args.command}")
    except (
        ConfigError,
        TokenizerError,
        DatasetError,
        TrainingError,
        LLMError,
        OSError,
        RuntimeError,
    ) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.command == "doctor" and report.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
