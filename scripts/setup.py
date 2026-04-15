# ===========================================================
# setup.py — Platform setup: Drive mount, repo clone, dirs
# ===========================================================
"""
Handles one-time setup tasks:
  - Mount Google Drive (Colab)
  - Clone/pull private GitHub repo
  - Create output directories
"""

import os
import subprocess
import sys

from scripts.config import PLATFORM, PATHS, REPO_BRANCH, REPO_OWNER, REPO_NAME


# ===========================================================
# GOOGLE DRIVE
# ===========================================================
def mount_drive():
    """Mount Google Drive on Colab."""
    if PLATFORM == 'colab':
        from google.colab import drive
        drive.mount('/content/drive')
        print("  Google Drive mounted")
    else:
        print("  Drive mount skipped (not Colab)")


# ===========================================================
# GITHUB TOKEN
# ===========================================================
def get_github_token():
    """Retrieve GitHub token from platform secrets."""
    try:
        if PLATFORM == 'colab':
            from google.colab import userdata
            return userdata.get('GITHUB_TOKEN') or ''
        elif PLATFORM == 'kaggle':
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret('GITHUB_TOKEN') or ''
    except Exception:
        pass
    return ''


# ===========================================================
# CLONE / PULL REPO
# ===========================================================
def clone_or_pull_repo():
    """Clone or pull the project repository (Colab/Kaggle only)."""
    if PLATFORM == 'local':
        print("  Repo clone skipped (local mode)")
        return None

    if PLATFORM == 'colab':
        repo_dir = '/content/SupportAI'
    else:
        repo_dir = '/kaggle/working/SupportAI'

    token = get_github_token()
    if token:
        repo_url = f'https://{token}@github.com/{REPO_OWNER}/{REPO_NAME}.git'
        print("  Private repo (token set)")
    else:
        repo_url = f'https://github.com/{REPO_OWNER}/{REPO_NAME}.git'
        print("  Public repo (no token)")

    if not os.path.exists(repo_dir):
        subprocess.run(
            ['git', 'clone', '-b', REPO_BRANCH, repo_url, repo_dir],
            check=True
        )
    else:
        subprocess.run(
            ['git', '-C', repo_dir, 'pull', 'origin', REPO_BRANCH],
            check=True
        )

    head = subprocess.check_output(
        ['git', '-C', repo_dir, 'rev-parse', '--short', 'HEAD'], text=True
    ).strip()
    print(f"  Repo: {head} ({REPO_BRANCH})")

    # Add to Python path
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    return repo_dir


# ===========================================================
# CREATE DIRECTORIES
# ===========================================================
def create_directories():
    """Create output directories if they don't exist."""
    for key in ['data_dir', 'model_dir', 'export_dir']:
        os.makedirs(PATHS[key], exist_ok=True)
    print(f"  Output dirs ready: {PATHS['drive_dir']}")


# ===========================================================
# ENTRY POINT
# ===========================================================
def run_setup():
    """Run full platform setup."""
    print("=" * 60)
    print("SETUP")
    print("=" * 60)
    mount_drive()
    clone_or_pull_repo()
    create_directories()
    print()
