"""Load environment configuration from the backend project directory."""

from pathlib import Path

from dotenv import load_dotenv


def load_backend_env(backend_dir: Path | None = None) -> None:
    project_dir = backend_dir or Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env", override=False)