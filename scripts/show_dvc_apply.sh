#!/usr/bin/env bash
# Task 1.4 (apply) evidence for the report: commits, pointer file, remote, sync status.   Run: bash scripts/show_dvc_apply.sh
cd "$HOME/23L0682_A2_Ops" || exit 1
. .venv/bin/activate
export DVC_NO_ANALYTICS=1
run() { printf '\n$ %s\n' "$*"; eval "$*"; }
echo "Task 1.4 apply evidence | user: $(whoami) | $(date '+%F %T') | project: 23L0682_A2_Ops"

run "git log --oneline --stat -3"
run "cat data/raw.dvc                         # the pointer file Git stores"
run "cat data/.gitignore                      # the real images are ignored by Git"
run "git remote -v"
run "dvc remote list                          # DagsHub storage (no credentials shown)"
run "du -sh data/raw && find data/raw -type f | wc -l   # size and number of images on disk"
run "dvc status --cloud                       # 'up to date' = DagsHub storage has every image"
