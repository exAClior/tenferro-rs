import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


def toolkit(path: Path, version: str) -> None:
    (path / "bin").mkdir(parents=True)
    (path / "lib64").mkdir()
    (path / "version.json").write_text(json.dumps({"cuda": {"version": version}}))
    compiler = path / "bin/nvcc"
    compiler.write_text('''#!/bin/bash
if [ "$1" = --version ]; then echo "fake nvcc for test"; exit 0; fi
while [ "$1" != -o ]; do shift; done
printf '.visible .entry tenferro_toolkit_smoke() {}\n' > "$2"
''')
    compiler.chmod(0o755)


class CudaToolkitCacheTests(unittest.TestCase):
    def test_copy_and_restore_exercise_cached_compiler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "installed", root / "cache"
            toolkit(source, "12.8.1")
            env = dict(os.environ, CUDA_PATH=str(source), GITHUB_ENV=str(root / "env"), GITHUB_PATH=str(root / "path"))
            command = ["bash", str(ROOT / "scripts/ci/install_cuda_toolkit_hosted.sh"), "12.8", str(cache)]
            subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            self.assertEqual((cache / "bin/nvcc").read_bytes(), (source / "bin/nvcc").read_bytes())
            self.assertIn(f"CUDA_PATH={cache}", (root / "env").read_text())
            # A restored matching cache wins over a newer host installation.
            (source / "version.json").write_text(json.dumps({"cuda": {"version": "13.0.0"}}))
            (source / "bin/nvcc").unlink()
            result = subprocess.run(command, env=env, check=True, capture_output=True, text=True)
            self.assertIn(f"Using CUDA toolkit at {cache}", result.stdout)
            self.assertIn("fake nvcc for test", result.stdout)

    def test_partial_cache_falls_back_to_valid_installation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, cache = root / "installed", root / "cache"
            toolkit(source, "12.8.1")
            toolkit(cache, "12.8.1")
            (cache / "bin/nvcc").unlink()
            env = dict(os.environ, CUDA_PATH=str(source), GITHUB_ENV=str(root / "env"), GITHUB_PATH=str(root / "path"))
            subprocess.run(["bash", str(ROOT / "scripts/ci/install_cuda_toolkit_hosted.sh"), "12.8", str(cache)], env=env, check=True, capture_output=True)
            self.assertTrue((cache / "bin/nvcc").is_file())


if __name__ == "__main__":
    unittest.main()
