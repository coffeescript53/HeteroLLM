#!/usr/bin/env python3
"""Upload this reviewed release to a NEW PRIVATE GitHub repository.

Uses the current gh account. Never changes an existing repository's visibility,
overwrites an existing remote, or enables Pages. Requires Python, Git, and gh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str, check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=ROOT, text=True, input=input,
                          capture_output=True, check=check)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="HeteroLLM", help="New repository name in the signed-in personal account")
    parser.add_argument("--check", action="store_true", help="Read-only readiness check; do not create, commit, or push")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.repo):
        raise SystemExit("Use a repository name containing letters, digits, '.', '_' or '-'.")
    for tool in ("git", "gh"):
        if shutil.which(tool) is None:
            raise SystemExit(f"Install {tool} on the computer used for upload, then rerun this command.")
    manifest = json.loads((ROOT / "PUBLIC_SOURCE_MANIFEST.json").read_text())
    paths = [entry["path"] for entry in manifest["files"]] + ["PUBLIC_SOURCE_MANIFEST.json"]
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or not path.is_file():
            raise SystemExit(f"Invalid release path: {entry['path']}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise SystemExit(f"Release file changed: {entry['path']}. Review the change before publishing this snapshot.")
    auth = run("gh", "auth", "status", "--hostname", "github.com", check=False)
    if auth.returncode:
        raise SystemExit("GitHub is not signed in. Run: gh auth login --hostname github.com\nThen rerun this script. No repository was created.")
    account = json.loads(run("gh", "api", "--hostname", "github.com", "user").stdout)
    owner = account["login"]
    target = f"{owner}/{args.repo}"
    identity = account.get("name") or owner
    email = f"{account['id']}+{owner}@users.noreply.github.com"
    existing = run("gh", "repo", "view", target, "--json", "nameWithOwner", check=False)
    if existing.returncode == 0:
        raise SystemExit(f"{target} already exists. Nothing was changed. Use --repo with a new name or manage the existing repository directly.")
    if (ROOT / ".git").exists():
        if run("git", "remote").stdout.strip():
            raise SystemExit("This checkout already has a remote. Nothing was changed.")
        indexed = set(run("git", "ls-files", "-z").stdout.split("\0")) - {""}
        if indexed - set(paths):
            raise SystemExit("The Git index contains files outside this release. Nothing was changed.")
    else:
        parent_git = run("git", "rev-parse", "--show-toplevel", check=False)
        if parent_git.returncode == 0:
            raise SystemExit("Extract this release outside another Git checkout before uploading.")
    print(f"Account: {owner}\nDestination: {target} (PRIVATE)\nCommit identity: {identity} <{email}>\nFiles: {len(paths)}")
    if args.check:
        print("Readiness check complete. No changes made.")
        return
    if not (ROOT / ".git").exists():
        run("git", "init", "-b", "main")
    if run("git", "branch", "--show-current").stdout.strip() != "main":
        raise SystemExit("The local branch is not main. Nothing was pushed.")
    run("git", "add", "--pathspec-from-file=-", "--pathspec-file-nul", input="\0".join(paths) + "\0")
    if run("git", "diff", "--cached", "--quiet", check=False).returncode:
        run("git", "-c", f"user.name={identity}", "-c", f"user.email={email}",
            "commit", "-m", "Prepare HeteroLLM systems engineering portfolio")
    run("gh", "repo", "create", target, "--private", "--description",
        "70B mixed-GPU inference measurement: C++ tracing, Python supervision, and reproducible benchmarks.")
    url = f"https://github.com/{target}"
    run("git", "remote", "add", "origin", url + ".git")
    push = run("git", "-c", "credential.helper=", "-c", "credential.helper=!gh auth git-credential",
               "push", "-u", "origin", "main", check=False)
    if push.returncode:
        raise SystemExit(f"Private repository created at {url}, but the push did not finish.\nYour local commit and origin remain. After checking GitHub authentication, run: git push -u origin main")
    state = json.loads(run("gh", "repo", "view", target, "--json", "url,visibility").stdout)
    print(f"Uploaded: {state['url']}\nVisibility: {state['visibility']}\nPages was not enabled. Keep this repository private until you choose to release it.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        print("A Git/GitHub operation failed. Existing state was preserved; no force push or deletion was attempted.", file=sys.stderr)
        print(error.stderr or str(error), file=sys.stderr)
        raise SystemExit(1)
