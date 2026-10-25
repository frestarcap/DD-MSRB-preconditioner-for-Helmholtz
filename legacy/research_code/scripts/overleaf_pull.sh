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

cd "$ROOT"

if [[ -n "$(git status --porcelain -- "$PAPER_DIR")" ]]; then
  echo "Refusing to overwrite local paper changes in $PAPER_DIR."
  echo "Commit/stash them first, or inspect with: git status --short -- $PAPER_DIR"
  exit 1
fi

mkdir -p "$(dirname "$MIRROR_DIR")"
if [[ ! -d "$MIRROR_DIR/.git" ]]; then
  git clone "$OVERLEAF_URL" "$MIRROR_DIR"
fi

git -C "$MIRROR_DIR" pull --rebase origin "$OVERLEAF_BRANCH"

REMOTE_DIR="$MIRROR_DIR"
if [[ -n "$REMOTE_SUBDIR" ]]; then
  REMOTE_DIR="$MIRROR_DIR/$REMOTE_SUBDIR"
fi

if [[ ! -d "$REMOTE_DIR" ]]; then
  echo "Overleaf source directory not found: $REMOTE_DIR"
  exit 1
fi

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
  "$REMOTE_DIR"/ "$PAPER_DIR"/

echo "Synced Overleaf -> $PAPER_DIR"
git status --short -- "$PAPER_DIR"
