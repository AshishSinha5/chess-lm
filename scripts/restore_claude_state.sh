#!/usr/bin/env bash
# Run once on a new pod, after Claude Code is installed. Links memory + history from the volume.
set -euo pipefail
SRC=/workspace/claude-home/projects
mkdir -p "$HOME/.claude"
[ -d "$SRC" ] || { echo "nothing to restore at $SRC"; exit 1; }
rm -rf "$HOME/.claude/projects"
ln -s "$SRC" "$HOME/.claude/projects"
echo "linked ~/.claude/projects -> $SRC"
