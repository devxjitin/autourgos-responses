"""
Regression test: importing autourgos_responses must not mutate process-wide
gRPC/TensorFlow/glog environment variables. Same root cause and fix as
autourgos-openaichat's configure_runtime_environment() (see that package's
identically-named test for the full rationale) -- this package re-exports
the same now-neutered function.
"""

import subprocess
import sys


def test_importing_package_does_not_set_grpc_tensorflow_env_vars():
    code = (
        "import os\n"
        "import autourgos_responses\n"
        "for var in ('GRPC_VERBOSITY', 'GLOG_minloglevel', 'TF_CPP_MIN_LOG_LEVEL'):\n"
        "    assert var not in os.environ, f'{var} was set: {os.environ[var]!r}'\n"
        "print('OK')\n"
    )
    env = {k: v for k, v in __import__("os").environ.items()
           if k not in ("GRPC_VERBOSITY", "GLOG_minloglevel", "TF_CPP_MIN_LOG_LEVEL")}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "OK" in result.stdout
