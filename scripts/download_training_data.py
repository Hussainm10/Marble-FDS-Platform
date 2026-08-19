#!/usr/bin/env python3
"""
Marble FDS — Training Data Downloader
=====================================

Automatically downloads the public ML training datasets from Kaggle.
Generic — works for any Marble FDS deployment.

Datasets:
- PaySim: Synthetic mobile money transaction data (6.3M transactions, 471MB)
- Elliptic: Bitcoin transaction graph data (203K transactions, 666MB)

Prerequisites:
1. Kaggle account (free): https://www.kaggle.com/
2. Kaggle API credentials

Usage:
    python scripts/download_training_data.py

The script will:
1. Check for Kaggle credentials
2. Download PaySim dataset
3. Download Elliptic dataset
4. Extract to correct directories
5. Verify file integrity
"""

import os
import sys
import json
import subprocess
import hashlib
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "bridge" / "training" / "data"

DATASETS = {
    "paysim": {
        "kaggle_dataset": "ealaxi/paysim1",
        "target_dir": TRAINING_DATA_DIR / "paysim",
        "expected_files": ["PS_20174392719_1491204439457_log.csv"],
        "description": "PaySim Mobile Money Simulator (6.3M transactions)",
    },
    "elliptic": {
        "kaggle_dataset": "ellipticco/elliptic-data-set",
        "target_dir": TRAINING_DATA_DIR / "elliptic",
        "expected_files": [
            "elliptic_txs_features.csv",
            "elliptic_txs_classes.csv",
            "elliptic_txs_edgelist.csv"
        ],
        "description": "Elliptic Bitcoin Transaction Graph (203K transactions)",
    }
}

KAGGLE_CREDENTIALS_PATH = Path.home() / ".kaggle" / "kaggle.json"


def print_header():
    """Print script header."""
    print("=" * 60)
    print("Marble FDS — Training Data Downloader")
    print("=" * 60)
    print()


def check_kaggle_credentials():
    """Check if Kaggle credentials are set up."""
    print("[1/4] Checking Kaggle credentials...")

    if KAGGLE_CREDENTIALS_PATH.exists():
        print(f"  ✓ Found credentials at {KAGGLE_CREDENTIALS_PATH}")
        return True

    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        print("  ✓ Found credentials in environment variables")
        return True

    print("  ✗ Kaggle credentials not found!")
    print()
    print("  To set up Kaggle credentials:")
    print("  1. Go to https://www.kaggle.com/settings")
    print("  2. Scroll to 'API' section")
    print("  3. Click 'Create New Token'")
    print("  4. Download kaggle.json")
    print("  5. Move it to ~/.kaggle/kaggle.json")
    print("  6. Run: chmod 600 ~/.kaggle/kaggle.json")
    print()
    print("  Or set environment variables:")
    print("    export KAGGLE_USERNAME=your_username")
    print("    export KAGGLE_KEY=your_api_key")
    print()
    return False


def install_kaggle_cli():
    """Install kaggle CLI if not present."""
    print("[2/4] Checking Kaggle CLI...")

    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  ✓ Kaggle CLI installed: {result.stdout.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("  → Installing Kaggle CLI...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "kaggle"],
            check=True,
            capture_output=True
        )
        print("  ✓ Kaggle CLI installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install Kaggle CLI: {e}")
        return False


def download_dataset(name: str, config: dict) -> bool:
    """Download a single dataset from Kaggle."""
    target_dir = config["target_dir"]
    kaggle_dataset = config["kaggle_dataset"]
    expected_files = config["expected_files"]

    print(f"\n  Downloading {config['description']}...")
    print(f"  Source: kaggle.com/datasets/{kaggle_dataset}")
    print(f"  Target: {target_dir}")

    # Check if already downloaded
    all_exist = all((target_dir / f).exists() for f in expected_files)
    if all_exist:
        print(f"  ✓ Already downloaded - skipping")
        return True

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download from Kaggle
    try:
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", kaggle_dataset,
                "-p", str(target_dir),
                "--unzip"
            ],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"  ✗ Download failed: {result.stderr}")
            return False

        # Handle nested directory (Elliptic extracts to subdirectory)
        if name == "elliptic":
            nested_dir = target_dir / "elliptic_bitcoin_dataset"
            if nested_dir.exists():
                for f in nested_dir.glob("*"):
                    f.rename(target_dir / f.name)
                nested_dir.rmdir()

        # Verify files exist
        missing = [f for f in expected_files if not (target_dir / f).exists()]
        if missing:
            print(f"  ✗ Missing files after download: {missing}")
            return False

        print(f"  ✓ Downloaded successfully")
        return True

    except subprocess.TimeoutExpired:
        print(f"  ✗ Download timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def download_all_datasets():
    """Download all required datasets."""
    print("[3/4] Downloading datasets...")

    success = True
    for name, config in DATASETS.items():
        if not download_dataset(name, config):
            success = False

    return success


def verify_downloads():
    """Verify all downloads are complete and correct."""
    print("\n[4/4] Verifying downloads...")

    all_good = True
    for name, config in DATASETS.items():
        target_dir = config["target_dir"]
        expected_files = config["expected_files"]

        print(f"\n  {name}:")
        for filename in expected_files:
            filepath = target_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / (1024 * 1024)
                print(f"    ✓ {filename} ({size_mb:.1f} MB)")
            else:
                print(f"    ✗ {filename} - MISSING")
                all_good = False

    return all_good


def create_symlink():
    """Create convenience symlink for paysim."""
    paysim_dir = DATASETS["paysim"]["target_dir"]
    source = paysim_dir / "PS_20174392719_1491204439457_log.csv"
    link = paysim_dir / "paysim_transactions.csv"

    if source.exists() and not link.exists():
        try:
            link.symlink_to(source.name)
            print("\n  ✓ Created symlink: paysim_transactions.csv")
        except OSError:
            # Windows might not support symlinks without admin
            pass


def print_summary(success: bool):
    """Print final summary."""
    print()
    print("=" * 60)
    if success:
        print("✓ All training data downloaded successfully!")
        print()
        print("Next steps:")
        print("  1. Set up .env file (copy from .env.example)")
        print("  2. Run: docker-compose up -d")
        print("  3. Or retrain models with:")
        print("     python bridge/training/scripts/train_marbel_v2.py")
        print("     python bridge/training/scripts/train_gnn_v3_hybrid.py")
    else:
        print("✗ Some downloads failed. Please check errors above.")
        print()
        print("Manual download:")
        print("  PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1")
        print("  Elliptic: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
    print("=" * 60)


def main():
    """Main entry point."""
    print_header()

    # Step 1: Check credentials
    if not check_kaggle_credentials():
        sys.exit(1)

    # Step 2: Install Kaggle CLI
    if not install_kaggle_cli():
        sys.exit(1)

    # Step 3: Download datasets
    if not download_all_datasets():
        print_summary(False)
        sys.exit(1)

    # Step 4: Verify
    success = verify_downloads()

    # Create convenience symlink
    if success:
        create_symlink()

    print_summary(success)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
