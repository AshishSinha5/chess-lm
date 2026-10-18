#!/usr/bin/env bash
# Copy Claude Code memory + conversation history to the persistent volume.
# Only ~/.claude/projects is copied. Login tokens (.credentials.json) are NOT copied.
set -euo pipefail
DEST=/workspace/claude-home
mkdir -p "$DEST/projects"
rsync -a --delete "$HOME/.claude/projects/" "$DEST/projects/"
echo "synced to $DEST/projects ($(du -sh "$DEST/projects" | cut -f1))"
