"""Commit generated recommendation HTML into GitHub via the Contents API.

Render's disk is ephemeral and the checkout is not a writable git remote, so a
local `git commit && git push` is unreliable. A fine-grained PAT with Contents
write access can create files over HTTPS.

Required env (set these in the Render dashboard):
  GITHUB_TOKEN   PAT with Contents: Read and write on this repo
  GITHUB_REPO    owner/name, e.g. slakshnarasimhan/betterhome

Optional:
  GITHUB_BRANCH            default: main
  GITHUB_GENERATED_PREFIX  default: web_app/generated

Commits include [skip render] so Render does not redeploy on every brochure.
Failures are logged and never raise — generation still succeeds for the user.
"""
from __future__ import annotations

import base64
import os
import re
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote

import requests

API_VERSION = "2022-11-28"
TIMEOUT_SECONDS = 60


def github_commit_enabled() -> bool:
    return bool(os.getenv("GITHUB_TOKEN") and os.getenv("GITHUB_REPO"))


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": API_VERSION,
    }


def _repo() -> str:
    repo = (os.getenv("GITHUB_REPO") or "").strip().strip("/")
    if repo.startswith("github.com/"):
        repo = repo[len("github.com/") :]
    if repo.endswith(".git"):
        repo = repo[:-4]
    return repo


def _branch() -> str:
    return (os.getenv("GITHUB_BRANCH") or "main").strip() or "main"


def _prefix() -> str:
    prefix = (os.getenv("GITHUB_GENERATED_PREFIX") or "web_app/generated").strip().strip("/")
    return prefix or "web_app/generated"


def _digits(value: Optional[str]) -> str:
    return "".join(ch for ch in str(value or "") if ch.isdigit()) or "unknown"


def _safe_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (cleaned or "recommendation")[:80]


def repo_path_for(local_path: str, phone: Optional[str] = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{_prefix()}/{_digits(phone)}/{ts}_{_safe_stem(local_path)}.html"


def _contents_url(repo_path: str) -> str:
    encoded = "/".join(quote(seg, safe="") for seg in repo_path.split("/") if seg)
    return f"https://api.github.com/repos/{_repo()}/contents/{encoded}"


def _existing_sha(repo_path: str) -> Optional[str]:
    response = requests.get(
        _contents_url(repo_path),
        headers=_headers(),
        params={"ref": _branch()},
        timeout=TIMEOUT_SECONDS,
    )
    if response.status_code == 404:
        return None
    if response.status_code >= 400:
        print(f"GitHub GET contents failed ({response.status_code}): {response.text[:500]}")
        return None
    sha = response.json().get("sha")
    return sha if isinstance(sha, str) else None


def commit_generated_html(local_path: str, phone: Optional[str] = None) -> bool:
    """Upload one local HTML file as a new GitHub commit. Returns True on success."""
    if not github_commit_enabled():
        print("GitHub commit skipped: set GITHUB_TOKEN and GITHUB_REPO on Render")
        return False
    if not local_path or not os.path.isfile(local_path):
        print(f"GitHub commit skipped: missing file {local_path}")
        return False

    size_bytes = os.path.getsize(local_path)
    # Contents API hard limit is 100 MB; stay well under it.
    if size_bytes > 80 * 1024 * 1024:
        print(f"GitHub commit skipped: {local_path} is {size_bytes} bytes")
        return False

    try:
        with open(local_path, "rb") as handle:
            content_b64 = base64.b64encode(handle.read()).decode("ascii")
        repo_path = repo_path_for(local_path, phone)
        sha = _existing_sha(repo_path)
        payload = {
            "message": f"Save generated recommendation {os.path.basename(repo_path)}\n\n[skip render]",
            "content": content_b64,
            "branch": _branch(),
        }
        if sha:
            payload["sha"] = sha
        response = requests.put(
            _contents_url(repo_path),
            headers=_headers(),
            json=payload,
            timeout=TIMEOUT_SECONDS,
        )
        if response.status_code in (200, 201):
            html_url = (response.json().get("content") or {}).get("html_url")
            print(f"GitHub commit succeeded: {html_url or repo_path}")
            return True
        print(f"GitHub commit failed ({response.status_code}): {response.text[:800]}")
        return False
    except Exception as exc:
        print(f"Warning: GitHub commit failed for {local_path}: {exc}")
        return False
