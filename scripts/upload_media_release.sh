#!/usr/bin/env bash
# Optional: publish the curated interaction media as a GitHub release (e.g. to later drop the
# two ~53 MB videos from git history-free checkouts). Uploads the committed files in
# docs/eval_results/interaction_media/ - nothing is re-run or regenerated.
#
# Requires the GitHub CLI with write access:  gh auth login
# Usage: bash scripts/upload_media_release.sh [tag]   (default tag: interaction-media-v1)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${GITHUB_REPO:-BharatVe/Learning-to-Interact-with-Web-Forms}"
TAG="${1:-interaction-media-v1}"
MEDIA="$ROOT_DIR/docs/eval_results/interaction_media"

command -v gh >/dev/null 2>&1 || { echo "[FAIL] GitHub CLI 'gh' not found" >&2; exit 1; }
shopt -s nullglob
assets=("$MEDIA"/*.webm "$MEDIA"/*.png)
[ "${#assets[@]}" -gt 0 ] || { echo "[FAIL] no media in $MEDIA" >&2; exit 1; }

gh release create "$TAG" --repo "$REPO" --title "Interaction media ${TAG#interaction-media-}" \
  --notes-file "$ROOT_DIR/docs/eval_results/INTERACTION_MEDIA.md" "${assets[@]}"
echo "[OK] assets on $TAG:"
gh release view "$TAG" --repo "$REPO" --json assets --jq '.assets[].name'
