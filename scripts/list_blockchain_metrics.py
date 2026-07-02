#!/usr/bin/env python3
"""List available blockchain metrics from Quantiacs.

This script fetches and displays all available blockchain metrics
that can be ingested into the database.
"""

import sys
import os
from pathlib import Path

# Load .env file BEFORE any imports
def _load_env():
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k not in os.environ:
                    os.environ[k] = v
    return project_root

# Set up paths
project_root = _load_env()
src_dir = project_root / "src"
sys.path.insert(0, str(src_dir))

# Ensure API_KEY is set
if not os.environ.get("API_KEY", "").strip():
    print("\nERROR: API_KEY is not set in .env file", file=sys.stderr)
    print(f"Add to {project_root}/.env:\n  API_KEY=your-key-here\n", file=sys.stderr)
    sys.exit(2)

try:
    import qnt.data as qndata
    import pandas as pd
except ImportError as e:
    print(f"\nERROR: Missing required package: {e}", file=sys.stderr)
    print("Install with: conda activate qntdev && pip install quantiacs-source::qnt", file=sys.stderr)
    sys.exit(1)


def main():
    """Fetch and display blockchain metrics."""
    print("=" * 80)
    print("Blockchain Metrics from Quantiacs")
    print("=" * 80)
    print()
    
    try:
        print("Fetching list of available blockchain metrics...")
        metrics = qndata.blockchaincom_load_list()
        
        if metrics is None:
            print("ERROR: No metrics returned from API")
            sys.exit(1)
        
        # Convert to DataFrame if it's not already
        if isinstance(metrics, pd.DataFrame):
            df = metrics
        elif hasattr(metrics, 'to_pandas'):
            df = metrics.to_pandas()
        else:
            # Try to convert xarray or other formats
            try:
                import xarray as xr
                if isinstance(metrics, xr.DataArray) or isinstance(metrics, xr.Dataset):
                    df = metrics.to_pandas()
                else:
                    # Try as list/dict
                    df = pd.DataFrame(metrics)
            except Exception as e:
                print(f"ERROR: Could not convert metrics to DataFrame: {e}")
                print(f"Type: {type(metrics)}")
                print(f"Value: {metrics}")
                sys.exit(1)
        
        print(f"\nFound {len(df)} blockchain metrics:\n")
        print(df.to_string(index=False))
        
        # Save to CSV for reference
        output_file = project_root / "scripts" / "blockchain_metrics_list.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Saved metrics list to: {output_file}")
        
        # Display summary
        print("\n" + "=" * 80)
        print("Summary:")
        print(f"  Total metrics: {len(df)}")
        if 'id' in df.columns:
            print(f"  Metric IDs: {', '.join(df['id'].astype(str).head(10).tolist())}")
            if len(df) > 10:
                print(f"  ... and {len(df) - 10} more")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nERROR: Failed to fetch blockchain metrics: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
