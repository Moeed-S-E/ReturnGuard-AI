#!/usr/bin/env bash
# ReturnGuard git bootstrap + per-task branching helper.
# Usage:
#   bash setup_git.sh init          # one-time: git init, create main + dev
#   bash setup_git.sh start <name>  # create & checkout task/<name> from dev
#   bash setup_git.sh finish <name> # merge task/<name> into dev (no-ff), delete branch
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

cmd="${1:-}"

case "$cmd" in
  init)
    [ -d .git ] || git init
    git symbolic-ref HEAD refs/heads/main 2>/dev/null || true
    git add .gitignore returnguard setup_git.sh *.md datasets 2>/dev/null || true
    if git rev-parse --verify HEAD >/dev/null 2>&1; then
      echo "Repo already has commits."
    else
      git commit -m "chore: project scaffolding and specs" || true
    fi
    git show-ref --verify --quiet refs/heads/dev || git branch dev
    echo "Done. main holds specs/scaffolding, dev is the integration branch."
    ;;
  start)
    name="${2:?usage: setup_git.sh start <name>}"
    git checkout dev
    git checkout -b "task/${name}"
    echo "On branch task/${name} (from dev)."
    ;;
  finish)
    name="${2:?usage: setup_git.sh finish <name>}"
    git add -A
    git commit -m "feat(${name}): implement task" || echo "nothing to commit"
    git checkout dev
    git merge --no-ff "task/${name}" -m "merge task/${name} into dev"
    git branch -d "task/${name}"
    echo "Merged task/${name} into dev."
    ;;
  *)
    echo "usage: bash setup_git.sh {init|start <name>|finish <name>}"
    exit 1
    ;;
esac
