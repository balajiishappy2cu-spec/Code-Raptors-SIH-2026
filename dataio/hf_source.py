"""Per-file access to the real Turing Synthetic Radar Dataset on Hugging Face.

The official helper ``turing_deinterleaving_challenge.download_dataset`` wraps
``snapshot_download`` with per-split allow patterns, which fetches an entire split
(thousands of files, tens of GB). The 24-hour MVP needs a few tens of files, so this
module lists the repository tree and downloads only the individual files the sampler
selects. Everything else about the data (layout, loader, metadata) is unchanged.

The dataset is gated: listing works anonymously, downloading needs a token. Put it in
``HUGGING_FACE_TOKEN`` (or ``HF_TOKEN``), or in a ``.env`` file at the repository root.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common.config import REPO_ROOT
from common.logging_utils import get_logger

LOGGER = get_logger(__name__)

_API_ROOT = "https://huggingface.co/api/datasets"
_RESOLVE_ROOT = "https://huggingface.co/datasets"
_TOKEN_ENV_VARS = ("HUGGING_FACE_TOKEN", "HF_TOKEN", "HUGGINGFACE_TOKEN")

#: TSRD subset directory naming, e.g. ``stare/train_stare/config_0.h5``.
_SUBSET_DIRS = {"train": "train", "validation": "val", "test": "test"}


class HuggingFaceUnavailable(RuntimeError):
    """Raised when the real dataset cannot be listed or downloaded."""


def read_hf_token() -> str | None:
    """Return a Hugging Face token from the environment or a ``.env`` file."""
    for var in _TOKEN_ENV_VARS:
        value = os.getenv(var)
        if value:
            return value.strip()
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*([A-Z_]+)\s*=\s*(.+)\s*$", line)
            if match and match.group(1) in _TOKEN_ENV_VARS:
                return match.group(2).strip().strip("'\"")
    return None


def _request(url: str, token: str | None, *, timeout: int = 60) -> Any:
    """Issue a GET request, adding the bearer token when one is available."""
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=timeout)


def split_dir(mode: str, subset: str) -> str:
    """Return the repository directory for a mode/subset pair.

    Args:
        mode: ``stare`` or ``scan``.
        subset: ``train``, ``validation`` or ``test``.

    Returns:
        For example ``stare/train_stare``.
    """
    if subset not in _SUBSET_DIRS:
        msg = f"Unknown subset {subset!r}; expected one of {sorted(_SUBSET_DIRS)}"
        raise ValueError(msg)
    return f"{mode}/{_SUBSET_DIRS[subset]}_{mode}"


def list_files(
    repo_id: str,
    directory: str,
    *,
    token: str | None = None,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    """List ``.h5`` files in one repository directory, following pagination.

    Args:
        repo_id: Hugging Face dataset repository id.
        directory: repository-relative directory, e.g. ``stare/test_stare``.
        token: optional Hugging Face token.
        timeout: per-request timeout in seconds.

    Returns:
        A list of ``{"path": ..., "size": ...}`` entries.

    Raises:
        HuggingFaceUnavailable: if the listing cannot be retrieved.
    """
    url = f"{_API_ROOT}/{repo_id}/tree/main/{directory}"
    entries: list[dict[str, Any]] = []
    seen_cursors: set[str] = set()
    while url:
        try:
            response = _request(url, token, timeout=timeout)
            payload = json.load(response)
            link_header = response.headers.get("Link", "")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            msg = f"Could not list {repo_id}:{directory} ({exc})"
            raise HuggingFaceUnavailable(msg) from exc

        for entry in payload:
            if entry.get("type") == "file" and str(entry.get("path", "")).endswith(".h5"):
                entries.append({"path": entry["path"], "size": int(entry.get("size", 0))})

        match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        url = match.group(1) if match else ""
        if url:
            if url in seen_cursors:  # pragma: no cover - defensive against loops
                break
            seen_cursors.add(url)
    return entries


def download_file(
    repo_id: str,
    repo_path: str,
    destination: str | Path,
    *,
    token: str | None = None,
    timeout: int = 300,
    overwrite: bool = False,
) -> Path:
    """Download one file from a Hugging Face dataset repository.

    Args:
        repo_id: dataset repository id.
        repo_path: repository-relative file path.
        destination: local file path to write.
        token: Hugging Face token (required for gated repositories).
        timeout: request timeout in seconds.
        overwrite: re-download even if the destination already exists.

    Returns:
        Path of the downloaded file.

    Raises:
        HuggingFaceUnavailable: if the download fails (including HTTP 401 when the
            repository is gated and no valid token was supplied).
    """
    out = Path(destination)
    if out.exists() and not overwrite and out.stat().st_size > 0:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{_RESOLVE_ROOT}/{repo_id}/resolve/main/{repo_path}"
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with _request(url, token, timeout=timeout) as response, tmp.open("wb") as handle:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        hint = ""
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
            hint = (
                " -- the TSRD is gated; set HUGGING_FACE_TOKEN in the environment or in "
                "a .env file at the repository root and accept the dataset terms on "
                "huggingface.co"
            )
        msg = f"Download failed for {repo_path} ({exc}){hint}"
        raise HuggingFaceUnavailable(msg) from exc
    tmp.replace(out)
    return out


def probe_availability(repo_id: str, mode: str, subset: str = "test") -> tuple[bool, str]:
    """Check whether the real dataset can actually be downloaded here.

    Performs a listing and a ranged read of the smallest file so that a gated
    repository is detected before the sampler commits to a long download.

    Args:
        repo_id: dataset repository id.
        mode: ``stare`` or ``scan``.
        subset: subset used for the probe.

    Returns:
        ``(available, reason)``.
    """
    token = read_hf_token()
    try:
        entries = list_files(repo_id, split_dir(mode, subset), token=token)
    except HuggingFaceUnavailable as exc:
        return False, str(exc)
    if not entries:
        return False, f"No .h5 files listed in {split_dir(mode, subset)}"

    smallest = min(entries, key=lambda entry: entry["size"])
    url = f"{_RESOLVE_ROOT}/{repo_id}/resolve/main/{smallest['path']}"
    request = urllib.request.Request(url)
    request.add_header("Range", "bytes=0-1023")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read(16)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        reason = f"Listing succeeded but download is not permitted ({exc})"
        if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403):
            reason += "; the TSRD is gated and no valid HUGGING_FACE_TOKEN was found"
        return False, reason
    return True, f"{len(entries)} files listed and readable in {split_dir(mode, subset)}"
