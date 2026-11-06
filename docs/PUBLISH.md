# GitHub and portfolio publication

The release is ready to upload. The user chose to keep it private until applying. No GitHub account is authenticated in the preparation environment, so no GitHub destination, push, or public URL is claimed.

## Create the private repository

On any computer with Python 3.10+, Git, and the [GitHub CLI](https://cli.github.com/), extract the finished source ZIP and open its `HeteroLLM` folder. This does not require the GPU workstation.

```bash
gh auth login --hostname github.com
python3 scripts/create-private-github.py
```

The script uses the signed-in personal account, creates a new **private** repository named `HeteroLLM`, commits only the verified release files using the account's public profile name and GitHub no-reply email, and pushes them. It prints the actual URL. It refuses an existing repository or remote, uses no force push, and does not enable Pages. `--repo AnotherName` chooses a different new name. `--check` performs a read-only readiness check.

The license and attribution files are already included. MIT describes reuse of the original contributions; it is separate from test results. No additional GPU experiment is required for this upload.

If this checkout already has a GitHub remote, manage that repository directly; the initial-upload helper deliberately refuses to replace it.

## Make it visible when applying

1. When ready, change the repository's visibility to public in GitHub's repository settings.
2. Under **Settings → Pages**, choose **GitHub Actions** as the build source.
3. Under **Actions → Publish portfolio**, run the workflow on `main`.
4. Copy the repository URL and the URL reported by the successful deployment. Put the actual links on the resume and verify them while signed out.

The Pages workflow is manual and runs only for a public repository. A normal source push runs CPU checks but does not publish the page. The page discovers its GitHub link from a conventional GitHub Pages project URL; no username is invented in the source.

The configured workflow follows [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages). The uploader uses the [GitHub CLI repository interface](https://cli.github.com/manual/gh_repo_create). Hosted workflow execution has not been verified until it runs in the connected account.

## Work entirely offline

```bash
python3 scripts/demo.py
python3 scripts/build-portfolio.py
```

Open `dist/index.html`. The build includes working local evidence links and `downloads/HeteroLLM-Source.zip`, so a reviewer can inspect the full source. The final delivery also provides a prebuilt offline portfolio ZIP. No server, model download, or graphics driver is required to read it.

## Release identity

`PUBLIC_SOURCE_MANIFEST.json` inventories this presentation release. `docs/release_history/` preserves the original licensed delivery's records. If files are changed after delivery, the initial-upload helper stops on the changed hash; review and update the release deliberately rather than bypassing that check.
