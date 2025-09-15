#!/usr/bin/env bash
set -euo pipefail

# ---- config ----
DEST="wim:~/edm2"                # scp target
DIST_DIR="dist"                 # local output dir for tarballs
MARKER_FILENAME="__COMMIT.txt"  # fixed marker filename at repo root in tarball

# ---- usage ----
if [ $# -lt 1 ]; then
  echo "Usage: $0 \"ship message here\""
  exit 2
fi
MSG="$1"

# ---- ensure git repo ----
git rev-parse --git-dir >/dev/null

# ---- commit ONLY tracked changes; abort if nothing to commit ----
if ! git commit -am "$MSG" ; then
  echo "No tracked changes to commit (or commit failed). Aborting."
  exit 1
fi

# ---- push changes; abort if failed ----
if ! git push ; then
  echo "Push failed. Aborting."
  exit 1
fi

# ---- capture metadata ----
HASH=$(git rev-parse --short=8 HEAD)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
STAMP=$(date +%Y%m%dT%H%M%S)   # lokalny czas
PROJECT=$(basename "$(git rev-parse --show-toplevel)")

# ---- build a clean export of EXACT commit (tracked files only) ----
TMPDIR=$(mktemp -d)
cleanup() { rm -rf "$TMPDIR"; }
trap cleanup EXIT

# UWAGA: bez --prefix -> pliki trafią bezpośrednio do TMPDIR (bez dodatkowego katalogu)
git archive --format=tar HEAD | tar -x -C "$TMPDIR"

# ---- write marker file at repo root inside the export ----
MARKER_PATH="${TMPDIR}/${MARKER_FILENAME}"
{
  echo "project:  ${PROJECT}"
  echo "commit:   ${HASH}"
  echo "branch:   ${BRANCH}"
  echo "datetime: ${STAMP}"
  echo "message:  ${MSG}"
} > "$MARKER_PATH"

# ---- pack tar.gz with files at top-level (no extra dir) ----
mkdir -p "$DIST_DIR"
TARBALL="${DIST_DIR}/${PROJECT}-${HASH}-${STAMP}.tar.gz"
# pack the whole TMPDIR
tar -C "$TMPDIR" -czf "$TARBALL" .

echo "Created: $TARBALL"

# ---- ship to remote ----
scp "$TARBALL" "$DEST"
echo "Shipped to: $DEST"