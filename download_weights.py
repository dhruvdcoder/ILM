#!/usr/bin/env python3
"""
Script to download model files from Google Drive using gdown.
Based on the file IDs extracted from rclone lsjson output.
"""

import os
import gdown
from pathlib import Path

# Dictionary of file IDs and their corresponding filenames
files_to_download = {
    "1sj6cHxPhXSy-EJWwY1WvYZA4dI-sdHQa": "ilm-lm1b.pt",
    "1_g_00HsM6wq49BzbOMaTa993KegMelxj": "ilm-star_hard.pt",
    "1UYL-mS3qIqTpRk-a4ClPt1LkJ5F4JkD3": "ilm-star_medium.pt",
    "1fu0JeZvweSxnAz5rcJ0O_hFUF-pHY2ib": "ilm-stories.pt",
    "1gNKscQeG41d_2KOHO8QnaAUnzowt02Kc": "ilm-zebra.pt",
    "1AuFx9AVXQycPdj-EfZg76kBg-VUvF7jG": "it-star_easy.pt",
    "1aY55IScuBSgC3H_jvmqF0OBIjcNXYFRf": "it-star_hard.pt",
    "14_tvrhtn9Ru_Q7QUK9A_kQN8dkZPtmVk": "it-star_medium.pt",
    "12hbyYYun-CQAKCv8pgnye2i8Eg03ku52": "mdm-lm1b.pt",
    "1Ku9BTdNolcA08J_PUtoPR5ktbVZILPyh": "mdm-star_easy.pt",
    "1IpQVQrsGsXzG92jKQ2hiuCd16o57C9zz": "mdm-star_hard.pt",
    "1_P_cjSgqVpwUCWmMD1Urk5_LK3i6XscM": "mdm-star_medium.pt",
    "1QNZV4eTKWg9UFiMqFiqCQBsaftzfmlR5": "mdm-stories.pt",
    "1XAQsvjG8Q2gbHysihBJDPIT-nTZU85pH": "mdm-zebra.pt",
    "1TTtXSQhZxNwmJZUbz9sAf7bOPwPi__mq": "xlnet-star_easy.pt",
    "1IO5P7loj0-_E04RsxwffpJHSRb8Zbfdw": "xlnet-star_hard.pt",
    "1SXXJ5BzW7lqb1VlU4Y3VILNb0qzb8Cmf": "xlnet-star_medium.pt",
}


def download_files(download_dir="./model_weights", skip_existing=True):
    """
    Download all files using gdown.

    Args:
        download_dir (str): Directory to save downloaded files
        skip_existing (bool): Skip download if file already exists
    """
    # Create download directory if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(files_to_download)} files to {download_dir}")
    print("=" * 60)

    for file_id, filename in files_to_download.items():
        model_dir = os.path.join(download_dir, filename.split(".")[0])
        output_path = os.path.join(model_dir, filename)

        # Skip if file exists and skip_existing is True
        if skip_existing and os.path.exists(output_path):
            print(f"Skipping {filename} (already exists)")
            continue

        print(f"Downloading {filename}...")

        try:
            # Download using gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            print(f"Successfully downloaded {filename}")

        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")

        print("-" * 40)

    print("Download process completed!")


def download_specific_files(file_patterns, download_dir="./model_weights"):
    """
    Download only files matching specific patterns.

    Args:
        file_patterns (list): List of patterns to match (e.g., ['ilm-', 'mdm-star'])
        download_dir (str): Directory to save downloaded files
    """
    matching_files = {}

    for file_id, filename in files_to_download.items():
        if any(pattern in filename for pattern in file_patterns):
            matching_files[file_id] = filename

    if not matching_files:
        print(f"No files found matching patterns: {file_patterns}")
        return

    print(
        f"Found {len(matching_files)} files matching patterns: {file_patterns}"
    )

    # Create download directory if it doesn't exist
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    for file_id, filename in matching_files.items():
        output_path = os.path.join(download_dir, filename)
        print(f"Downloading {filename}...")

        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            print(f"Successfully downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {str(e)}")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Download model files from Google Drive using gdown"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="model_weights",
        help="Directory to download files to (default: model_weights)",
    )
    parser.add_argument(
        "patterns",
        nargs="*",
        help="File patterns to match for selective downloading (e.g., 'ilm-', 'star')",
    )

    args = parser.parse_args()

    if args.patterns:
        # Download specific files based on patterns
        print(f"Downloading files matching patterns: {args.patterns}")
        download_specific_files(args.patterns, download_dir=args.output_dir)
    else:
        # Download all files
        download_files(download_dir=args.output_dir)

    print("\nAvailable files:")
    for filename in sorted(files_to_download.values()):
        print(f"  - {filename}")
