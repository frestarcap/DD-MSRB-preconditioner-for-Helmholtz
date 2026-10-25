#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="${OVERLEAF_PROJECT:-JSC}"
if [[ "${1:-}" == "JSC" || "${1:-}" == "JCP" ]]; then
  PROJECT="$1"
  shift
fi
case "$PROJECT" in
  JSC)
    DEFAULT_MIRROR=".overleaf/paper"
    DEFAULT_SUBDIR="LaTeX_DL_468198_240419"
    DEFAULT_URL="<overleaf-project-url>"
    DEFAULT_PAPER_DIR="JSC"
    ;;
  JCP)
    DEFAULT_MIRROR=".overleaf/jcp"
    DEFAULT_SUBDIR=""
    DEFAULT_URL="<overleaf-project-url>"
    # Since 2026-10-24 the JCP Overleaf project tracks the CAS manuscript;
    # JCP/ (elsarticle) is an archived copy that is no longer synced.
    DEFAULT_PAPER_DIR="JCP_CAS"
    ;;
  *) echo "Unknown project: $PROJECT (use JSC or JCP)" >&2; exit 1 ;;
esac
PAPER_DIR="${OVERLEAF_PAPER_DIR:-$DEFAULT_PAPER_DIR}"
MIRROR_DIR="${OVERLEAF_MIRROR_DIR:-$DEFAULT_MIRROR}"
REMOTE_SUBDIR="${OVERLEAF_REMOTE_SUBDIR-$DEFAULT_SUBDIR}"
OVERLEAF_URL="${OVERLEAF_URL:-$DEFAULT_URL}"
OVERLEAF_BRANCH="${OVERLEAF_BRANCH:-main}"
COMMIT_MESSAGE="${1:-Sync local manuscript to Overleaf}"

cd "$ROOT"

[[ -d "$PAPER_DIR" ]] || { echo "Paper directory not found: $PAPER_DIR" >&2; exit 1; }

mkdir -p "$(dirname "$MIRROR_DIR")"
if [[ ! -d "$MIRROR_DIR/.git" ]]; then
  git clone "$OVERLEAF_URL" "$MIRROR_DIR"
fi

if [[ -n "$(git -C "$MIRROR_DIR" status --porcelain)" ]]; then
  echo "Overleaf mirror has uncommitted changes. Inspect: git -C $MIRROR_DIR status"
  exit 1
fi

REMOTE_DIR="$MIRROR_DIR"
if [[ -n "$REMOTE_SUBDIR" ]]; then
  REMOTE_DIR="$MIRROR_DIR/$REMOTE_SUBDIR"
fi
mkdir -p "$REMOTE_DIR"

rsync -a --delete \
  --exclude='.git/' \
  --exclude='/template.pdf' \
  --exclude='/elsarticle-template-*.pdf' \
  --exclude='/manuscript.pdf' \
  --exclude='*.abs' \
  --exclude='.DS_Store' \
  --exclude='.vscode/' \
  --exclude='*.aux' \
  --exclude='*.spl' \
  --exclude='*.log' \
  --exclude='*.out' \
  --exclude='*.toc' \
  --exclude='*.synctex.gz' \
  --exclude='*.fls' \
  --exclude='*.fdb_latexmk' \
  --exclude='*.bbl' \
  --exclude='*.blg' \
  --exclude='*.bcf' \
  --exclude='*.run.xml' \
  "$PAPER_DIR"/ "$REMOTE_DIR"/

git -C "$MIRROR_DIR" add -A
if git -C "$MIRROR_DIR" diff --cached --quiet; then
  echo "No Overleaf changes to commit."
else
  git -C "$MIRROR_DIR" commit -m "$COMMIT_MESSAGE"
fi

git -C "$MIRROR_DIR" pull --rebase origin "$OVERLEAF_BRANCH"
git -C "$MIRROR_DIR" push origin "$OVERLEAF_BRANCH"

echo "Synced $PAPER_DIR -> Overleaf"
