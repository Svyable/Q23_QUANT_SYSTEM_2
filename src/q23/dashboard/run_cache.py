"""
Ephemeral Run Cache

Manages temporary strategy runs that are automatically deleted when the
dashboard session ends. This enables PMs to experiment with strategy
variations without cluttering the permanent output directories.

Usage:
    cache = EphemeralRunCache()
    
    # Get directory for a new ephemeral run
    run_dir = cache.get_run_dir("test_momentum_v2")
    
    # List cached runs
    runs = cache.list_runs()
    
    # Cleanup happens automatically on process exit
    # Or manually:
    cache.cleanup()
"""

from __future__ import annotations

import atexit
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# Default cache location
DEFAULT_CACHE_DIR = Path(".cache/ephemeral_runs")


class EphemeralRunCache:
    """
    Manage temporary strategy runs that are deleted on app close.
    
    Ephemeral runs are stored in a cache directory and automatically
    cleaned up when the application exits. This allows PMs to experiment
    freely without worrying about disk space or clutter.
    
    Attributes:
        cache_dir: Path to the ephemeral runs cache directory
    """
    
    _instance: Optional["EphemeralRunCache"] = None
    
    def __new__(cls, cache_dir: Optional[Path] = None) -> "EphemeralRunCache":
        """Singleton pattern to ensure only one cache manager exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the ephemeral run cache.
        
        Args:
            cache_dir: Custom cache directory (defaults to .cache/ephemeral_runs)
        """
        if getattr(self, '_initialized', False):
            return
        
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
        
        self._initialized = True
    
    def get_run_dir(self, run_id: str) -> Path:
        """
        Get directory for an ephemeral run.
        
        Creates the directory if it doesn't exist. The run_id is sanitized
        to be a valid directory name.
        
        Args:
            run_id: Identifier for the run (will be sanitized)
            
        Returns:
            Path to the run directory
        """
        # Sanitize run_id
        safe_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_id)
        safe_id = safe_id.strip("_") or "unnamed_run"
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        full_id = f"{safe_id}_{timestamp}"
        
        run_dir = self.cache_dir / full_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        meta_path = run_dir / "_ephemeral_meta.json"
        import json
        meta = {
            "run_id": run_id,
            "created": datetime.now().isoformat(),
            "ephemeral": True,
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        
        return run_dir
    
    def list_runs(self) -> List[str]:
        """
        List all cached ephemeral runs.
        
        Returns:
            List of run directory names
        """
        if not self.cache_dir.exists():
            return []
        
        return [d.name for d in self.cache_dir.iterdir() if d.is_dir()]
    
    def get_run_metadata(self, run_name: str) -> Optional[dict]:
        """
        Get metadata for an ephemeral run.
        
        Args:
            run_name: Name of the run directory
            
        Returns:
            Metadata dict or None if not found
        """
        import json
        
        run_dir = self.cache_dir / run_name
        meta_path = run_dir / "_ephemeral_meta.json"
        
        if meta_path.exists():
            try:
                return json.loads(meta_path.read_text())
            except Exception:
                pass
        
        return None
    
    def delete_run(self, run_name: str) -> bool:
        """
        Delete a specific ephemeral run.
        
        Args:
            run_name: Name of the run directory to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        run_dir = self.cache_dir / run_name
        
        if run_dir.exists() and run_dir.is_dir():
            try:
                shutil.rmtree(run_dir)
                return True
            except Exception:
                pass
        
        return False
    
    def cleanup(self) -> int:
        """
        Delete all ephemeral runs.
        
        This is automatically called on process exit.
        
        Returns:
            Number of runs deleted
        """
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        try:
            runs = self.list_runs()
            for run_name in runs:
                if self.delete_run(run_name):
                    count += 1
            
            # Remove the cache directory itself if empty
            if self.cache_dir.exists() and not list(self.cache_dir.iterdir()):
                self.cache_dir.rmdir()
                
        except Exception:
            pass
        
        return count
    
    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """
        Delete ephemeral runs older than max_age_hours.
        
        Useful for cleaning up runs from crashed sessions.
        
        Args:
            max_age_hours: Maximum age in hours before deletion
            
        Returns:
            Number of runs deleted
        """
        from datetime import timedelta
        
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        
        for run_name in self.list_runs():
            meta = self.get_run_metadata(run_name)
            if meta and "created" in meta:
                try:
                    created = datetime.fromisoformat(meta["created"])
                    if created < cutoff:
                        if self.delete_run(run_name):
                            count += 1
                except Exception:
                    pass
        
        return count
    
    @property
    def size_bytes(self) -> int:
        """Get total size of cached runs in bytes."""
        if not self.cache_dir.exists():
            return 0
        
        total = 0
        for f in self.cache_dir.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        
        return total
    
    @property
    def size_mb(self) -> float:
        """Get total size of cached runs in MB."""
        return self.size_bytes / (1024 * 1024)


# Convenience function to get singleton instance
def get_run_cache() -> EphemeralRunCache:
    """Get the singleton EphemeralRunCache instance."""
    return EphemeralRunCache()
