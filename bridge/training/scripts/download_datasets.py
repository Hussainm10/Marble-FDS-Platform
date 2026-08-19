"""Download PaySim and Elliptic datasets from Kaggle.

Usage:
    python download_datasets.py

Prerequisites:
    1. Create a Kaggle account at https://www.kaggle.com
    2. Go to Account Settings -> API -> Create New Token
    3. This downloads kaggle.json with your credentials
    4. Either:
       a) Place kaggle.json in ~/.kaggle/ (Linux/Mac) or C:/Users/<user>/.kaggle/ (Windows)
       b) Or set KAGGLE_USERNAME and KAGGLE_KEY environment variables
"""

import os
import sys
import zipfile
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
PAYSIM_DIR = DATA_DIR / "paysim"
ELLIPTIC_DIR = DATA_DIR / "elliptic"


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured."""
    # Check environment variables
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        print("[OK] Kaggle credentials found in environment variables")
        return True

    # Check kaggle.json file
    kaggle_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path("C:/Users") / os.getenv("USERNAME", "") / ".kaggle" / "kaggle.json",
    ]

    for path in kaggle_paths:
        if path.exists():
            print(f"[OK] Kaggle credentials found at {path}")
            return True

    print("[ERROR] Kaggle credentials not found!")
    print("\nTo set up Kaggle credentials:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Click 'Create New Token' under API section")
    print("3. Save kaggle.json to ~/.kaggle/ or set environment variables:")
    print("   export KAGGLE_USERNAME='your_username'")
    print("   export KAGGLE_KEY='your_api_key'")
    return False


def download_paysim():
    """Download PaySim dataset from Kaggle."""
    print("\n" + "=" * 60)
    print("Downloading PaySim Dataset")
    print("Source: https://www.kaggle.com/datasets/ealaxi/paysim1")
    print("=" * 60)

    PAYSIM_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    csv_file = PAYSIM_DIR / "PS_20174392719_1491204439457_log.csv"
    if csv_file.exists():
        size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"[OK] PaySim already downloaded ({size_mb:.1f} MB)")
        return True

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print("Downloading... (this may take a few minutes)")
        api.dataset_download_files(
            "ealaxi/paysim1",
            path=str(PAYSIM_DIR),
            unzip=True,
        )

        # Verify download
        if csv_file.exists():
            size_mb = csv_file.stat().st_size / (1024 * 1024)
            print(f"[OK] PaySim downloaded successfully ({size_mb:.1f} MB)")

            # Count rows
            with open(csv_file, "r") as f:
                row_count = sum(1 for _ in f) - 1  # Subtract header
            print(f"[OK] PaySim contains {row_count:,} transactions")
            return True
        else:
            print("[ERROR] Download completed but CSV file not found")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to download PaySim: {e}")
        return False


def download_elliptic():
    """Download Elliptic Bitcoin dataset from Kaggle."""
    print("\n" + "=" * 60)
    print("Downloading Elliptic Bitcoin Dataset")
    print("Source: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    print("=" * 60)

    ELLIPTIC_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    features_file = ELLIPTIC_DIR / "elliptic_txs_features.csv"
    edges_file = ELLIPTIC_DIR / "elliptic_txs_edgelist.csv"
    classes_file = ELLIPTIC_DIR / "elliptic_txs_classes.csv"

    if features_file.exists() and edges_file.exists() and classes_file.exists():
        print("[OK] Elliptic already downloaded")
        return True

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print("Downloading... (this may take a minute)")
        api.dataset_download_files(
            "ellipticco/elliptic-data-set",
            path=str(ELLIPTIC_DIR),
            unzip=True,
        )

        # Verify download
        if features_file.exists():
            # Count nodes
            with open(features_file, "r") as f:
                node_count = sum(1 for _ in f)
            print(f"[OK] Elliptic downloaded: {node_count:,} nodes")

            # Count edges
            if edges_file.exists():
                with open(edges_file, "r") as f:
                    edge_count = sum(1 for _ in f) - 1
                print(f"[OK] Elliptic edges: {edge_count:,}")

            return True
        else:
            print("[ERROR] Download completed but files not found")
            return False

    except Exception as e:
        print(f"[ERROR] Failed to download Elliptic: {e}")
        return False


def verify_datasets():
    """Verify both datasets are present and valid."""
    print("\n" + "=" * 60)
    print("Dataset Verification")
    print("=" * 60)

    # PaySim verification
    paysim_csv = PAYSIM_DIR / "PS_20174392719_1491204439457_log.csv"
    if paysim_csv.exists():
        import pandas as pd
        df = pd.read_csv(paysim_csv, nrows=5)
        print(f"\nPaySim columns: {list(df.columns)}")
        print(f"PaySim sample transaction types: {df['type'].unique().tolist()}")
        fraud_count = pd.read_csv(paysim_csv, usecols=["isFraud"])["isFraud"].sum()
        print(f"PaySim fraud transactions: {fraud_count:,}")
    else:
        print("[MISSING] PaySim dataset")

    # Elliptic verification
    elliptic_classes = ELLIPTIC_DIR / "elliptic_txs_classes.csv"
    if elliptic_classes.exists():
        import pandas as pd
        df = pd.read_csv(elliptic_classes)
        print(f"\nElliptic class distribution:")
        print(df["class"].value_counts())
    else:
        print("[MISSING] Elliptic dataset")


def main():
    print("=" * 60)
    print("FDS ML Training - Dataset Download")
    print("=" * 60)

    # Check credentials first
    if not check_kaggle_credentials():
        sys.exit(1)

    # Download datasets
    paysim_ok = download_paysim()
    elliptic_ok = download_elliptic()

    # Verify
    if paysim_ok and elliptic_ok:
        verify_datasets()
        print("\n" + "=" * 60)
        print("ALL DATASETS DOWNLOADED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nData location: {DATA_DIR}")
        print("Ready for Phase 3 (Train Marbel) and Phase 4 (Train GNN)")
    else:
        print("\n[ERROR] Some datasets failed to download")
        sys.exit(1)


if __name__ == "__main__":
    main()
