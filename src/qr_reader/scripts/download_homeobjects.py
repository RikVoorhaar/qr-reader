"""
Download the HomeObjects-3K dataset into the `data/` directory.

Dataset source:
    https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip

Usage:
    python src/qr_reader/scripts/download_homeobjects.py
"""

import os
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

URL = (
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/homeobjects-3K.zip"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
ZIP_PATH = DATA_DIR / "homeobjects-3K.zip"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Zip already downloaded: {ZIP_PATH}")
    else:
        print(f"Downloading {URL} → {ZIP_PATH}")
        urlretrieve(URL, str(ZIP_PATH))
        print("Download complete.")

    # Extract if not already extracted (the zip ships `images/`, `labels/`,
    # and a YAML metadata file directly).
    extracted_marker = DATA_DIR / "images"
    if extracted_marker.exists():
        print(f"Already extracted (found {extracted_marker})")
        return

    print(f"Extracting {ZIP_PATH} → {DATA_DIR}")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)
    print("Extraction complete.")


if __name__ == "__main__":
    main()
