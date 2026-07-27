#!/usr/bin/env bash
#
# Purge site/data imagery from the whole git history.
#
# WHY
#   Until the accompanying workflow change, every 10-minute run appended a
#   commit full of freshly rendered PNGs.  Git cannot delta-compress one
#   render against the next, so each commit added its images to the repository
#   permanently.  The result: 15,359 of the repository's 15,379 commits are
#   automated image snapshots, and the repository outgrew its GitHub quota.
#
#   The workflow now amends its previous snapshot instead of stacking a new
#   one, so growth is capped going forward.  This script reclaims what was
#   already written, by dropping site/data from every historical commit.
#
# WHAT SURVIVES
#   Every hand-written commit, with its message, author and date.  Only the
#   site/data paths are removed.  Image-only commits become empty and are
#   pruned, which is where essentially all the space comes back.
#
# WHAT BREAKS
#   Every commit SHA changes.  This is a history rewrite: existing clones and
#   forks must be re-cloned, and open PRs from before the rewrite will show
#   nonsense diffs.  Coordinate with anyone else working on the repo first.
#
# WHY NOT RUN FROM A CLOUD AGENT SESSION
#   git-filter-repo streams every blob in history through fast-export, so it
#   needs a complete (non-shallow, non-blobless) clone.  Fetching those ~4+ GB
#   of image blobs, and later pushing the rewritten history back, needs a
#   direct connection to GitHub.  Run this from a normal workstation.
#
# USAGE
#   ./tools/purge-image-history.sh <owner/repo> [--keep-current-images]
#
#   --keep-current-images  Re-commit the current contents of site/data after
#                          the purge, so the site keeps rendering and the
#                          animation frame buffers stay full.  Without it the
#                          site has no imagery until the next scheduled run,
#                          and all products restart with a single frame.
#
set -euo pipefail

REPO="${1:-}"
KEEP_IMAGES="${2:-}"

if [ -z "$REPO" ]; then
    echo "usage: $0 <owner/repo> [--keep-current-images]" >&2
    exit 2
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
    echo "git-filter-repo is required: pip install git-filter-repo" >&2
    exit 2
fi

WORKDIR="$(mktemp -d)"
CLONE="$WORKDIR/$(basename "$REPO")"

echo "==> Making a complete fresh clone (this downloads the full history)"
# filter-repo requires a full clone: no --depth, no --filter.
git clone "https://github.com/$REPO" "$CLONE"
cd "$CLONE"

# Track every remote branch locally, otherwise filter-repo leaves them alone
# and they go on pinning the old image blobs.
for ref in $(git branch -r | grep -v HEAD | sed 's#origin/##' | tr -d ' '); do
    git rev-parse --verify -q "$ref" >/dev/null || \
        git branch --track "$ref" "origin/$ref" >/dev/null 2>&1 || true
done

BEFORE="$(du -sh .git | cut -f1)"
echo "==> Size before: $BEFORE   commits: $(git rev-list --count --all)"

if [ "$KEEP_IMAGES" = "--keep-current-images" ]; then
    echo "==> Stashing the current imagery to restore after the rewrite"
    mkdir -p "$WORKDIR/data_backup"
    if [ -d site/data ]; then cp -a site/data/. "$WORKDIR/data_backup/"; fi
fi

echo "==> Rewriting history to drop site/data"
git filter-repo --path site/data --invert-paths --force

AFTER="$(du -sh .git | cut -f1)"
echo "==> Size after:  $AFTER   commits: $(git rev-list --count --all)"

# filter-repo deliberately drops the remote so a rewrite can't be pushed by
# accident. Put it back.
git remote add origin "https://github.com/$REPO"

if [ "$KEEP_IMAGES" = "--keep-current-images" ] && [ -n "$(ls -A "$WORKDIR/data_backup" 2>/dev/null)" ]; then
    echo "==> Restoring the current generation of imagery as one fresh commit"
    DEFAULT_BRANCH="$(git symbolic-ref --short HEAD)"
    git checkout -q "$DEFAULT_BRANCH"
    mkdir -p site/data
    cp -a "$WORKDIR/data_backup/." site/data/
    git add site/data
    git -c user.name="GitHub Actions Bot" \
        -c user.email="github-actions[bot]@users.noreply.github.com" \
        commit -q -m "chore: update satellite images [$(date -u '+%Y-%m-%d %H:%M UTC')]"
fi

cat <<EOF

================================================================
Rewrite complete, locally.  Nothing has been pushed yet.
  before: $BEFORE
  after:  $AFTER
  clone:  $CLONE

Review it, then push every rewritten branch:

  cd $CLONE
  git push --force --all origin

Two things to know about the GitHub side:

  * The size shown for the repository will NOT drop immediately.  The old
    objects stay until GitHub runs gc on their side.  Open a GitHub Support
    ticket asking them to run gc on $REPO once the force-push has landed.

  * Pull request refs (refs/pull/N/head) are permanent and are not rewritten
    by a force-push, so any PR that contains image commits will keep pinning
    those old blobs.  Support can address this along with the gc.

Then have everyone re-clone; old clones cannot be fast-forwarded onto the
rewritten history.
================================================================
EOF
