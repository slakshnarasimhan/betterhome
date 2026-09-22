"""Commit generated recommendation files into GitHub as soon as they exist.

Render cannot `git push` from the service checkout. After each successful
generation we create blobs and one commit on the target branch via the Git
Data API (HTML + PDF + Excel together).

Required on Render:
  GITHUB_TOKEN   PAT with Contents: Read and write

Optional:
  GITHUB_REPO              default slakshnarasimhan/betterhome
  GITHUB_BRANCH            default main
  GITHUB_GENERATED_PREFIX  default web_app/generated

Commits include [skip render] so Render does not redeploy on every brochure.
Failures are logged and never raise.
"""
from __future__ import annotations

import base64
import os
import re
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence, Tuple

import requests

API_VERSION = "2022-11-28"
TIMEOUT_SECONDS = 90
DEFAULT_REPO = "slakshnarasimhan/betterhome"
MAX_BLOB_BYTES = 80 * 1024 * 1024


def github_commit_enabled() -> bool:
    return bool((os.getenv("GITHUB_TOKEN") or "").strip())


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def _repo() -> str:
    repo = (
        os.getenv("GITHUB_REPO")
        or os.getenv("GITHUB_REPOSITORY")
        or DEFAULT_REPO
    ).strip().strip("/")
    if repo.startswith("github.com/"):
        repo = repo[len("github.com/") :]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo or DEFAULT_REPO


def _branch() -> str:
    return (os.getenv("GITHUB_BRANCH") or "main").strip() or "main"


def _prefix() -> str:
    prefix = (os.getenv("GITHUB_GENERATED_PREFIX") or "web_app/generated").strip().strip("/")
    return prefix or "web_app/generated"


def _digits(value: Optional[str]) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit()) or "unknown"


def _safe_name(path: str) -> str:
    name = os.path.basename(path)
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "recommendation.bin"


def dest_folder_for(phone: Optional[str] = None, timestamp: Optional[str] = None) -> str:
    ts = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{_prefix()}/{_digits(phone)}/{ts}"


def _api(method: str, path: str, payload: Optional[dict] = None) -> requests.Response:
    url = f"https://api.github.com/repos/{_repo()}{path}"
    return requests.request(
        method,
        url,
        headers=_headers(),
        json=payload,
        timeout=TIMEOUT_SECONDS,
    )


def _create_blob(raw: bytes) -> Optional[str]:
    response = _api(
        "POST",
        "/git/blobs",
        {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"},
    )
    if response.status_code not in (200, 201):
        print(f"GitHub blob create failed ({response.status_code}): {response.text[:800]}")
        return None
    sha = response.json().get("sha")
    return sha if isinstance(sha, str) else None


def _head_commit() -> Tuple[Optional[str], Optional[str]]:
    ref = _api("GET", f"/git/ref/heads/{_branch()}")
    if ref.status_code == 404:
        # Some tokens need the refs API with encoded branch names.
        ref = _api("GET", f"/git/matching-refs/heads/{_branch()}")
        if ref.status_code == 200 and isinstance(ref.json(), list) and ref.json():
            commit_sha = (ref.json()[0].get("object") or {}).get("sha")
        else:
            print(f"GitHub ref lookup failed ({ref.status_code}): {ref.text[:500]}")
            return None, None
    elif ref.status_code >= 400:
        print(f"GitHub ref lookup failed ({ref.status_code}): {ref.text[:500]}")
        return None, None
    else:
        commit_sha = (ref.json().get("object") or {}).get("sha")
    if not commit_sha:
        return None, None
    commit = _api("GET", f"/git/commits/{commit_sha}")
    if commit.status_code >= 400:
        print(f"GitHub commit lookup failed ({commit.status_code}): {commit.text[:500]}")
        return commit_sha, None
    tree_sha = (commit.json().get("tree") or {}).get("sha")
    return commit_sha, tree_sha


def commit_generated_files(local_paths: Sequence[str], phone: Optional[str] = None) -> bool:
    """Commit every existing local file in one GitHub commit. Returns True on success."""
    if not github_commit_enabled():
        print("GitHub commit skipped: set GITHUB_TOKEN on Render (Contents: Read and write)")
        return False

    existing: List[str] = []
    for path in local_paths or []:
        if path and os.path.isfile(path):
            existing.append(path)
        elif path:
            print(f"GitHub commit skipped missing file: {path}")
    if not existing:
        print("GitHub commit skipped: no generated files to upload")
        return False

    folder = dest_folder_for(phone)
    tree_entries = []
    for path in existing:
        size_bytes = os.path.getsize(path)
        if size_bytes > MAX_BLOB_BYTES:
            print(f"GitHub commit skipped {path}: {size_bytes} bytes exceeds blob limit")
            continue
        try:
            with open(path, "rb") as handle:
                raw = handle.read()
        except OSError as exc:
            print(f"GitHub commit could not read {path}: {exc}")
            continue
        blob_sha = _create_blob(raw)
        if not blob_sha:
            continue
        repo_path = f"{folder}/{_safe_name(path)}"
        tree_entries.append({
            "path": repo_path,
            "mode": "100644",
            "type": "blob",
            "sha": blob_sha,
        })
        print(f"GitHub blob ready: {repo_path} ({size_bytes} bytes)")

    if not tree_entries:
        return False

    parent_sha, base_tree = _head_commit()
    if not parent_sha or not base_tree:
        print("GitHub commit failed: could not read current branch head")
        return False

    tree_resp = _api("POST", "/git/trees", {"base_tree": base_tree, "tree": tree_entries})
    if tree_resp.status_code not in (200, 201):
        print(f"GitHub tree create failed ({tree_resp.status_code}): {tree_resp.text[:800]}")
        return False
    tree_sha = tree_resp.json().get("sha")
    names = ", ".join(os.path.basename(entry["path"]) for entry in tree_entries)
    commit_resp = _api(
        "POST",
        "/git/commits",
        {
            "message": f"Save generated recommendation {folder}\n\nFiles: {names}\n\n[skip render]",
            "tree": tree_sha,
            "parents": [parent_sha],
        },
    )
    if commit_resp.status_code not in (200, 201):
        print(f"GitHub commit create failed ({commit_resp.status_code}): {commit_resp.text[:800]}")
        return False
    new_sha = commit_resp.json().get("sha")
    ref_resp = _api("PATCH", f"/git/refs/heads/{_branch()}", {"sha": new_sha})
    if ref_resp.status_code not in (200, 201):
        print(f"GitHub branch update failed ({ref_resp.status_code}): {ref_resp.text[:800]}")
        return False
    print(
        f"GitHub commit succeeded: https://github.com/{_repo()}/tree/{_branch()}/{folder} ({new_sha})"
    )
    return True


def commit_generated_html(local_path: str, phone: Optional[str] = None) -> bool:
    return commit_generated_files([local_path], phone=phone)
