# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run the disaggregated prefill shell demo from the active conda env."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _runtime_env(runtime_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    script_dir = runtime_dir.parent.parent
    rpc_dir = runtime_dir / "rpc"
    rpc_socket_dir = script_dir / ".r"

    host_ip = env.get("VLLM_HOST_IP", "")
    if not host_ip or host_ip == "localhost":
        env["VLLM_HOST_IP"] = "127.0.0.1"
    elif host_ip != "127.0.0.1":
        raise SystemExit(
            "VLLM_HOST_IP must stay on loopback for this example. Use 127.0.0.1."
        )

    env["TMPDIR"] = str(runtime_dir / "tmp")
    env["VLLM_RPC_BASE_PATH"] = str(rpc_socket_dir)
    env["VLLM_CACHE_ROOT"] = str(runtime_dir / "cache")
    env["VLLM_CONFIG_ROOT"] = str(runtime_dir / "config")
    env["HF_HOME"] = str(runtime_dir / "hf")
    env["XDG_CACHE_HOME"] = str(runtime_dir / "xdg-cache")

    for key in (
        "TMPDIR",
        "VLLM_CACHE_ROOT",
        "VLLM_CONFIG_ROOT",
        "HF_HOME",
        "XDG_CACHE_HOME",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    rpc_dir.mkdir(parents=True, exist_ok=True)
    if rpc_socket_dir.is_symlink() or rpc_socket_dir.is_file():
        rpc_socket_dir.unlink()
    elif rpc_socket_dir.exists():
        shutil.rmtree(rpc_socket_dir)
    rpc_socket_dir.symlink_to(rpc_dir)

    return env


def _cleanup_runtime_link(script_dir: Path) -> None:
    rpc_socket_dir = script_dir / ".r"
    if rpc_socket_dir.is_symlink() or rpc_socket_dir.is_file():
        rpc_socket_dir.unlink()
    elif rpc_socket_dir.exists():
        shutil.rmtree(rpc_socket_dir)


def main() -> int:
    if os.environ.get("CONDA_DEFAULT_ENV") != "vllm":
        print(
            "Activate conda environment 'vllm' before running this script.",
            file=sys.stderr,
        )
        return 1

    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    repo_root = script_path.parents[2]
    shell_script = script_dir / "disaggregated_prefill.sh"

    runtime_dir = script_dir / ".runtime" / "disaggregated_prefill"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    env = _runtime_env(runtime_dir)
    try:
        completed = subprocess.run(
            ["bash", str(shell_script)],
            cwd=repo_root,
            env=env,
            check=False,
        )
    finally:
        _cleanup_runtime_link(script_dir)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
