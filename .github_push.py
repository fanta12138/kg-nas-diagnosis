#!/usr/bin/env python
"""Incremental push to GitHub over the REST Git Data API.

Why this exists
---------------
The local network blocks `github.com`'s smart-HTTP endpoint
(`CONNECT tunnel failed, response 502`), so `git push` cannot run here. It does
allow `api.github.com`, so we push through the documented Git Data API instead:
blob -> tree -> commit -> ref update.

Usage
-----
    export GH_TOKEN=$(gh auth token)          # `repo` scope
    python .github_push.py

The token is read from the environment and never written to disk.

Notes
-----
* Only blobs the remote does not already have are uploaded, so a re-run after a
  failure uploads the remainder rather than starting over.
* A flat tree of ~1600 entries intermittently returns 502 from the API; the
  retry loop below absorbs that (it succeeded on the first retry in practice).
"""

import base64
import http.client
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

OWNER = "fanta12138"
REPO = "kg-nas-diagnosis"
BRANCH = "master"
API = "https://api.github.com"
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

TOKEN = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
if not TOKEN:
    sys.exit("Set GH_TOKEN (repo scope) before running.")


def api(method, path, payload=None, retries=6):
    url = path if path.startswith("http") else f"{API}{path}"
    data = json.dumps(payload).encode() if payload is not None else None
    last = None
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {TOKEN}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        if data:
            req.add_header("Content-Type", "application/json")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:400]
            last = f"{method} {url} -> {e.code}\n{detail}"
            # 5xx and 429 are worth retrying; other 4xx are not.
            if e.code < 500 and e.code != 429:
                raise SystemExit(last)
        except (
            urllib.error.URLError,
            TimeoutError,
            ConnectionError,
            http.client.HTTPException,
            json.JSONDecodeError,
        ) as e:
            last = f"{method} {url} -> {e!r}"
        print(f"  retry {attempt}/{retries}: {last.splitlines()[0]}", flush=True)
        time.sleep(2 * attempt)
    raise SystemExit(last)


def git(*args):
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def main():
    head_sha = git("rev-parse", "HEAD").strip()
    ref = api("GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}")
    remote_commit = ref["object"]["sha"]
    if remote_commit == head_sha:
        print("Remote already at", head_sha[:8], "- nothing to push.")
        return

    remote_tree_sha = api(
        "GET", f"/repos/{OWNER}/{REPO}/git/commits/{remote_commit}"
    )["tree"]["sha"]
    remote_blobs = {
        e["path"]: e["sha"]
        for e in api(
            "GET", f"/repos/{OWNER}/{REPO}/git/trees/{remote_tree_sha}?recursive=1"
        )["tree"]
        if e["type"] == "blob"
    }
    print(f"remote {remote_commit[:8]}: {len(remote_blobs)} blobs")

    # One `ls-tree` call rather than 1500+ `rev-parse` calls: on Windows each
    # subprocess costs ~0.4 s, which would turn this into minutes.
    blob_map = {}
    for line in git(
        "ls-tree", "-r", "HEAD", "--format=%(objectname) %(path)"
    ).splitlines():
        sha, _, path = line.partition(" ")
        blob_map[path] = sha
    print(f"local HEAD: {len(blob_map)} files")

    uploaded = reused = 0
    for path, sha in sorted(blob_map.items()):
        if remote_blobs.get(path) == sha:
            reused += 1
            continue
        with open(os.path.join(REPO_ROOT, path), "rb") as fh:
            raw = fh.read()
        res = api(
            "POST",
            f"/repos/{OWNER}/{REPO}/git/blobs",
            {"content": base64.b64encode(raw).decode("ascii"), "encoding": "base64"},
        )
        if res["sha"] != sha:
            print(f"  ! {path}: server {res['sha'][:8]} != local {sha[:8]}")
        uploaded += 1
        if uploaded % 25 == 0:
            print(f"  ... {uploaded} uploaded", flush=True)
    print(f"blobs: {uploaded} uploaded, {reused} reused")

    tree = api(
        "POST",
        f"/repos/{OWNER}/{REPO}/git/trees",
        {
            "tree": [
                {"path": p, "mode": "100644", "type": "blob", "sha": s}
                for p, s in sorted(blob_map.items())
            ]
        },
    )
    print("tree:", tree["sha"][:8])

    commit = api(
        "POST",
        f"/repos/{OWNER}/{REPO}/git/commits",
        {
            "message": git("log", "-1", "--pretty=%B").strip(),
            "tree": tree["sha"],
            "parents": [remote_commit],
        },
    )
    print("commit:", commit["sha"][:8])

    api(
        "PATCH",
        f"/repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        {"sha": commit["sha"], "force": False},
    )
    print(f"pushed {BRANCH} -> {commit['sha'][:8]}")


if __name__ == "__main__":
    main()
