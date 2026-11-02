"""Tests for loading backend-local environment configuration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend_env import load_backend_env  # noqa: E402


class BackendEnvTests(unittest.TestCase):
    def test_loads_from_project_dir_and_preserves_process_value(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            backend_dir = temp_path / "backend"
            backend_dir.mkdir()
            (backend_dir / ".env").write_text(
                "GROQ_API_KEY=from_env_file\nOTHER_ENV_VALUE=loaded\n",
                encoding="utf-8",
            )
            working_dir = temp_path / "elsewhere"
            working_dir.mkdir()

            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("GROQ_API_KEY", None)
                os.environ.pop("OTHER_ENV_VALUE", None)
                old_cwd = Path.cwd()
                try:
                    os.chdir(working_dir)
                    load_backend_env(backend_dir)
                    self.assertEqual(os.environ["GROQ_API_KEY"], "from_env_file")
                    self.assertEqual(os.environ["OTHER_ENV_VALUE"], "loaded")

                    os.environ["GROQ_API_KEY"] = "from_process_environment"
                    load_backend_env(backend_dir)
                    self.assertEqual(
                        os.environ["GROQ_API_KEY"], "from_process_environment"
                    )
                finally:
                    os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()