from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import pandas as pd

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
except ImportError:  # Optional until user installs Drive deps.
    Request = None
    Credentials = None
    InstalledAppFlow = None
    build = None
    MediaIoBaseDownload = None


DEFAULT_SCOPES = ("https://www.googleapis.com/auth/drive.readonly",)
TRIAL_INDEX_COLUMNS = [
    "task_name",
    "trial_id",
    "video_capture1",
    "video_capture2",
    "kinematics_path",
    "transcription_path",
    "data_source",
]


@dataclass
class FileRecord:
    name: str
    path: str
    kind: str
    task_name: str
    trial_id: str


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_trial_id(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"capture[ _-]?[12]\b", "", stem)
    stem = re.sub(r"(kinematics?|transcription|transcript|gesture)s?\b", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem or "unknown_trial"


def _matches_keywords(filename: str, keywords: Sequence[str]) -> bool:
    lower = filename.lower()
    return any(keyword.lower() in lower for keyword in keywords)


def _classify_video_capture(filename: str) -> str:
    lower = filename.lower()
    if re.search(r"capture[ _-]?2\b", lower):
        return "video_capture2"
    if re.search(r"capture[ _-]?1\b", lower):
        return "video_capture1"
    return "video_capture1"


def _rows_to_df(rows: list[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in TRIAL_INDEX_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    if df.empty:
        return pd.DataFrame(columns=TRIAL_INDEX_COLUMNS)
    df = df[TRIAL_INDEX_COLUMNS].fillna("")
    return df.sort_values(["task_name", "trial_id"]).reset_index(drop=True)


def _iter_local_records(
    data_root: Path,
    video_extensions: Sequence[str],
    kinematics_keywords: Sequence[str],
    transcription_keywords: Sequence[str],
    tasks_filter: set[str] | None,
) -> Iterable[FileRecord]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {root}")

    ext_set = {ext.lower() for ext in video_extensions}
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        task_name = task_dir.name.strip()
        task_key = _normalize_name(task_name)
        if tasks_filter and task_key not in tasks_filter:
            continue
        for file_path in task_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(task_dir).as_posix().lower()
            filename = file_path.name
            suffix = file_path.suffix.lower()
            trial_id = _extract_trial_id(filename)
            if suffix in ext_set and "video/" in rel:
                kind = _classify_video_capture(filename)
            elif "kinematics/" in rel and _matches_keywords(filename, kinematics_keywords):
                kind = "kinematics_path"
            elif "transcription/" in rel and _matches_keywords(filename, transcription_keywords):
                kind = "transcription_path"
            else:
                continue
            yield FileRecord(
                name=filename,
                path=str(file_path.resolve()),
                kind=kind,
                task_name=task_name,
                trial_id=trial_id,
            )


def _load_drive_credentials(gdrive_cfg: Dict[str, Any]) -> Any:
    if Request is None or Credentials is None or InstalledAppFlow is None or build is None:
        raise ImportError(
            "Google Drive dependencies are missing. Install "
            "'google-api-python-client google-auth-httplib2 google-auth-oauthlib'."
        )

    credentials_path = Path(gdrive_cfg.get("credentials_path", "secrets/google_oauth_client.json"))
    token_path = Path(gdrive_cfg.get("token_path", "secrets/google_token.json"))
    scopes = tuple(gdrive_cfg.get("scopes", list(DEFAULT_SCOPES)))
    if not credentials_path.exists():
        raise FileNotFoundError(
            f"OAuth client credentials not found: {credentials_path}. "
            "Set ingestion.gdrive.credentials_path."
        )

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), scopes=scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(str(credentials_path), scopes=scopes)
            oauth_flow = str(gdrive_cfg.get("oauth_flow", "console")).lower()
            if oauth_flow == "local_server":
                creds = flow.run_local_server(port=0)
            else:
                creds = flow.run_local_server(port=0, open_browser=False)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def _list_drive_children(service: Any, folder_id: str) -> list[Dict[str, str]]:
    query = f"'{folder_id}' in parents and trashed=false"
    fields = "nextPageToken, files(id, name, mimeType, parents)"
    page_token = None
    items: list[Dict[str, str]] = []
    while True:
        response = (
            service.files()
            .list(q=query, fields=fields, pageToken=page_token, includeItemsFromAllDrives=True, supportsAllDrives=True)
            .execute()
        )
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items


def _download_drive_file(service: Any, file_id: str, destination: Path) -> Path:
    if MediaIoBaseDownload is None:
        raise ImportError(
            "Google Drive download dependency is missing. Install "
            "'google-api-python-client'."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination

    import io

    request = service.files().get_media(fileId=file_id)
    with destination.open("wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return destination


def _resolve_drive_root(service: Any, gdrive_cfg: Dict[str, Any], data_root: str) -> str:
    root_id = gdrive_cfg.get("root_folder_id")
    if root_id:
        return str(root_id)

    target_name = str(gdrive_cfg.get("jigsaw_root_name", Path(data_root).name or "jigsaw")).strip()
    children = _list_drive_children(service, "root")
    for item in children:
        if item.get("mimeType") == "application/vnd.google-apps.folder" and item.get("name", "").lower() == target_name.lower():
            return str(item["id"])
    raise ValueError(
        f"Could not find Drive root folder named '{target_name}'. "
        "Set ingestion.gdrive.root_folder_id to avoid name matching."
    )


def _iter_drive_records(
    data_root: str,
    video_extensions: Sequence[str],
    kinematics_keywords: Sequence[str],
    transcription_keywords: Sequence[str],
    tasks_filter: set[str] | None,
    gdrive_cfg: Dict[str, Any],
) -> Iterable[FileRecord]:
    creds = _load_drive_credentials(gdrive_cfg)
    service = build("drive", "v3", credentials=creds)
    root_folder_id = _resolve_drive_root(service, gdrive_cfg, data_root)
    ext_set = {ext.lower() for ext in video_extensions}
    materialize_local = bool(gdrive_cfg.get("materialize_local", True))
    cache_dir = Path(gdrive_cfg.get("cache_dir", "./data/gdrive_cache"))

    task_folders = [
        item
        for item in _list_drive_children(service, root_folder_id)
        if item.get("mimeType") == "application/vnd.google-apps.folder"
    ]
    for task in sorted(task_folders, key=lambda f: f["name"].lower()):
        task_name = task["name"].strip()
        task_key = _normalize_name(task_name)
        if tasks_filter and task_key not in tasks_filter:
            continue

        modality_folders = {
            item["name"].lower(): item["id"]
            for item in _list_drive_children(service, task["id"])
            if item.get("mimeType") == "application/vnd.google-apps.folder"
        }

        for modality_name, kind_hint in (
            ("video", "video"),
            ("kinematics", "kinematics"),
            ("transcription", "transcription"),
        ):
            folder_id = modality_folders.get(modality_name)
            if not folder_id:
                continue
            for file_item in _list_drive_children(service, folder_id):
                if file_item.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                filename = file_item["name"]
                suffix = Path(filename).suffix.lower()
                trial_id = _extract_trial_id(filename)
                if kind_hint == "video":
                    if suffix not in ext_set:
                        continue
                    kind = _classify_video_capture(filename)
                elif kind_hint == "kinematics":
                    if not _matches_keywords(filename, kinematics_keywords):
                        continue
                    kind = "kinematics_path"
                else:
                    if not _matches_keywords(filename, transcription_keywords):
                        continue
                    kind = "transcription_path"
                if materialize_local:
                    destination = cache_dir / _normalize_name(task_name) / modality_name / filename
                    resolved_path = str(_download_drive_file(service, file_item["id"], destination).resolve())
                else:
                    resolved_path = f"gdrive://{file_item['id']}/{filename}"
                yield FileRecord(
                    name=filename,
                    path=resolved_path,
                    kind=kind,
                    task_name=task_name,
                    trial_id=trial_id,
                )


def discover_trials(
    data_root: str | Path,
    video_extensions: Sequence[str] = (".avi", ".mp4", ".mov"),
    kinematics_keywords: Sequence[str] = ("kinematics",),
    transcription_keywords: Sequence[str] = ("transcription",),
    ingestion_config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Discover multimodal trial files and emit a normalized trial index DataFrame.

    With ingestion_config.source='gdrive', this uses OAuth + Drive API.
    """
    ingestion_config = ingestion_config or {}
    source = str(ingestion_config.get("source", "gdrive")).lower()
    tasks = ingestion_config.get("tasks", [])
    tasks_filter = {_normalize_name(t) for t in tasks} if tasks else None
    gdrive_cfg = dict(ingestion_config.get("gdrive", {}))

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    if source == "gdrive":
        records = _iter_drive_records(
            data_root=str(data_root),
            video_extensions=video_extensions,
            kinematics_keywords=kinematics_keywords,
            transcription_keywords=transcription_keywords,
            tasks_filter=tasks_filter,
            gdrive_cfg=gdrive_cfg,
        )
        data_source = "gdrive"
    elif source == "local":
        records = _iter_local_records(
            data_root=Path(data_root),
            video_extensions=video_extensions,
            kinematics_keywords=kinematics_keywords,
            transcription_keywords=transcription_keywords,
            tasks_filter=tasks_filter,
        )
        data_source = "local"
    elif source == "auto":
        try:
            records = _iter_drive_records(
                data_root=str(data_root),
                video_extensions=video_extensions,
                kinematics_keywords=kinematics_keywords,
                transcription_keywords=transcription_keywords,
                tasks_filter=tasks_filter,
                gdrive_cfg=gdrive_cfg,
            )
            data_source = "gdrive"
        except Exception:
            records = _iter_local_records(
                data_root=Path(data_root),
                video_extensions=video_extensions,
                kinematics_keywords=kinematics_keywords,
                transcription_keywords=transcription_keywords,
                tasks_filter=tasks_filter,
            )
            data_source = "local"
    else:
        raise ValueError("ingestion.source must be one of: gdrive, local, auto")

    for record in records:
        key = (_normalize_name(record.task_name), record.trial_id)
        row = grouped.setdefault(
            key,
            {
                "task_name": record.task_name,
                "trial_id": record.trial_id,
                "video_capture1": "",
                "video_capture2": "",
                "kinematics_path": "",
                "transcription_path": "",
                "data_source": data_source,
            },
        )
        if not row.get(record.kind):
            row[record.kind] = record.path

    rows = [
        row
        for row in grouped.values()
        if row.get("video_capture1") or row.get("video_capture2")
    ]
    return _rows_to_df(rows)


def save_trial_index(df: pd.DataFrame, output_path: str | Path) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalized = _rows_to_df(df.to_dict(orient="records") if not df.empty else [])
    normalized.to_csv(out, index=False)


def save_trial_discovery_report(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Optional helper for quick manual QA of discovered trials.
    """
    counts = {
        "num_trials": int(len(df)),
        "num_tasks": int(df["task_name"].nunique() if not df.empty else 0),
        "missing_kinematics": int((df["kinematics_path"] == "").sum() if not df.empty else 0),
        "missing_transcription": int((df["transcription_path"] == "").sum() if not df.empty else 0),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(counts, indent=2), encoding="utf-8")
