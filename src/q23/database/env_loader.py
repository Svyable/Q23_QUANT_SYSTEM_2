"""q23.database.env_loader

Environment variable loading for database scripts.
Ensures .env file is loaded and API keys are available.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


def load_env_file(project_root: Optional[Path] = None) -> None:
    """Load .env file into environment variables.
    
    Args:
        project_root: Project root directory (defaults to finding from script location)
    """
    if project_root is None:
        # Try to find project root by looking for .env file
        # Start from current working directory and walk up
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            env_file = parent / ".env"
            if env_file.exists():
                project_root = parent
                break
        
        if project_root is None:
            # Fallback: assume we're in scripts/ or src/ directory
            script_dir = Path(__file__).resolve().parent
            # Go up to find project root (from src/q23/database/ -> project root)
            project_root = script_dir.parent.parent.parent
    
    env_path = project_root / ".env"
    
    if not env_path.exists():
        return  # No .env file, skip
    
    # Simple .env parser (no external dependencies)
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        # Only set if not already in environment (env vars take precedence)
        if k not in os.environ:
            os.environ[k] = v


def ensure_api_key() -> None:
    """Ensure Quantiacs API_KEY is set.
    
    Raises:
        SystemExit: If API_KEY is not set
    """
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        print(
            "\nERROR: API_KEY is not set.\n"
            "Put it in .env file as:\n"
            "  API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx\n",
            file=sys.stderr,
        )
        sys.exit(2)


def setup_environment(project_root: Optional[Path] = None) -> Path:
    """Set up environment for database scripts.
    
    - Loads .env file
    - Ensures API_KEY is set
    - Sets PYTHONPATH if needed
    
    Args:
        project_root: Project root directory (auto-detected if None)
        
    Returns:
        Project root path
    """
    if project_root is None:
        # Try to find project root
        current = Path.cwd()
        for parent in [current] + list(current.parents):
            if (parent / ".env").exists() or (parent / "src" / "q23").exists():
                project_root = parent
                break
        
        if project_root is None:
            # Fallback
            script_dir = Path(__file__).resolve().parent
            project_root = script_dir.parent.parent.parent
    
    # Load .env file
    load_env_file(project_root)
    
    # Ensure API key is set
    ensure_api_key()
    
    # Set PYTHONPATH if src directory exists
    src_dir = project_root / "src"
    if src_dir.exists():
        pythonpath = os.environ.get("PYTHONPATH", "")
        if str(src_dir) not in pythonpath:
            os.environ["PYTHONPATH"] = f"{src_dir}:{pythonpath}" if pythonpath else str(src_dir)
    
    # Change to project root for consistent behavior
    os.chdir(project_root)
    
    return project_root
