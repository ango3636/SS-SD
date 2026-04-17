"""Download the full JIGSAWS dataset from Google Drive to the local cache.

Recursively mirrors the Drive folder structure under ``data/gdrive_cache/``,
pulling kinematics, transcriptions, metafiles, and experimental_setup files
that the SD training pipeline requires (videos are already cached).

Uses the same OAuth credentials as the rest of the pipeline
(``secrets/google_oauth_client.json``).

Usage::

    python scripts/download_jigsaws.py
    python scripts/download_jigsaws.py --dry-run   # list files without downloading
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import yaml
from suturing_pipeline.data.loader import (
    _download_drive_file,
    _list_drive_children,
    _load_drive_credentials,
    _resolve_drive_root,
)

FOLDER_MIME = "application/vnd.google-apps.folder"


def _walk_drive(service, folder_id: str, local_dir: Path, dry_run: bool) -> int:
    """Recursively download every file under *folder_id* into *local_dir*."""
    downloaded = 0
    children = _list_drive_children(service, folder_id)

    for item in sorted(children, key=lambda c: c.get("name", "").lower()):
        name = item["name"]
        child_id = item["id"]

        if item.get("mimeType") == FOLDER_MIME:
            downloaded += _walk_drive(
                service, child_id, local_dir / name, dry_run
            )
        else:
            dest = local_dir / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            if dry_run:
                print(f"  [dry-run] would download: {dest}")
            else:
                print(f"  downloading: {dest}")
                _download_drive_file(service, child_id, dest)
            downloaded += 1

    return downloaded


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full JIGSAWS dataset from Google Drive."
    )
    parser.add_argument(
        "--config",
        default="configs/base.yaml",
        help="Path to project config (for Drive credentials).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Override local destination directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files that would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    gdrive_cfg = cfg.get("ingestion", {}).get("gdrive", {})
    cache_dir = Path(args.cache_dir or gdrive_cfg.get("cache_dir", "./data/gdrive_cache"))
    data_root = cfg.get("paths", {}).get("data_root", "jigsaw")

    from googleapiclient.discovery import build

    print("Authenticating with Google Drive ...")
    creds = _load_drive_credentials(gdrive_cfg)
    service = build("drive", "v3", credentials=creds)

    root_id = _resolve_drive_root(service, gdrive_cfg, data_root)
    print(f"Drive root folder ID: {root_id}")
    print(f"Local cache dir:      {cache_dir.resolve()}")
    print()

    count = _walk_drive(service, root_id, cache_dir, dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n{count} file(s) would be downloaded.")
    else:
        print(f"\nDone. {count} new file(s) downloaded to {cache_dir.resolve()}")
        print(f"\nYour data_root for training is:\n  {cache_dir.resolve()}")


if __name__ == "__main__":
    main()
