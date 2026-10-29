"""Command-line tools for configuration inspection and environment diagnostics.

``doctor`` checks the stage-1 scaffold. It deliberately does not claim that a
model is ready: inference is implemented in a later stage. Missing torch and
model artifacts are reported as informational, not as scaffold failures.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from atlasai import __version__
from atlasai.config import DEFAULT_CONFIG_PATH, AtlasConfig, ConfigError, load_config

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
            else "Optional at stage 1; needed for later model stages",
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
                "File exists; contents are not validated at stage 1"
                if exists
                else "Not created yet; expected before training/inference",
            )
        )

    cpu_count = os.cpu_count()
    memory_mib = _linux_available_memory_mib()
    if cpu_count is not None and config.runtime.num_threads > cpu_count:
        checks.append(
            _check("threads", "warning", "Configured threads exceed visible logical CPUs")
        )
    if memory_mib is not None and memory_mib < 512:
        checks.append(_check("memory", "warning", "Less than 512 MiB available on the host"))

    statuses = {check["status"] for check in checks}
    status = "error" if "error" in statuses else "warning" if "warning" in statuses else "ok"
    return {
        "stage": 1,
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
        "model_status": "not_implemented",
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlasai", description="Atlas AI stage-1 scaffold")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="TOML config path (default: configs/atlas.toml, relative to the current directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("config", parents=[common], help="Print the validated config as JSON")
    commands.add_parser("doctor", parents=[common], help="Inspect the stage-1 environment as JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config_path = Path(args.config).expanduser().resolve(strict=False)
        config = load_config(config_path)
        report = (
            doctor_report(config, config_path) if args.command == "doctor" else config.to_dict()
        )
    except (ConfigError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.command == "doctor" and report.get("status") == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
