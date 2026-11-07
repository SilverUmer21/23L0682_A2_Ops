#!/usr/bin/env bash
# Task 1.4 - TOY PART: Git + DVC on a tiny CSV, in a NEW EMPTY FOLDER outside the project.
# Run:  bash scripts/lab_dvc_toy.sh        (safe to re-run: it only ever deletes its own throwaway folders)
# Every command is printed with a leading "$" before it runs, so one terminal screenshot shows what was done.
set -euo pipefail

LAB="$HOME/dvc_toy_lab"                    # the new empty folder
REMOTE="/mnt/k/dvc_remotes/dvc_toy_lab"    # remote storage = a plain folder on the K: drive
MARK="$HOME/.dvc_toy_lab.marker"           # proves the folders below were created by this script
. "$HOME/23L0682_A2_Ops/.venv/bin/activate"   # DVC is installed in the project venv
export DVC_NO_ANALYTICS=1                     # never send DVC's anonymous usage reports

run() { printf '\n$ %s\n' "$*"; eval "$*"; }
banner() { printf '\n========== %s ==========\n' "$*"; }

echo "Task 1.4 toy lab | user: $(whoami) | host: $(hostname) | $(date '+%F %T') | project: 23L0682_A2_Ops"

# --- start from a really empty folder (only ever removes folders this script created) ---
if [ -e "$LAB" ] || [ -e "$REMOTE" ]; then
  [ -f "$MARK" ] || { echo "Refusing: $LAB or $REMOTE already exist and were not created by this script."; exit 1; }
  rm -rf "$LAB" "$REMOTE"
fi
touch "$MARK"

banner "1. New empty folder, initialise Git and DVC"
run "mkdir -p $LAB && cd $LAB && ls -A | wc -l   # 0 = empty"
cd "$LAB"
run "git init -b main"
run "dvc init"
run "dvc config core.analytics false             # no anonymous usage reports"
run "git add -A && git commit -q -m 'Initialize Git and DVC'"

banner "2. Small CSV, version 1, tracked by DVC"
run "printf 'id,age,income,approved\n1,25,30000,0\n2,32,52000,1\n3,41,61000,1\n4,29,38000,0\n5,37,47000,1\n' > data.csv"
run "dvc add data.csv"
run "git add data.csv.dvc .gitignore && git commit -q -m 'v1: add data.csv (5 rows) tracked by DVC'"
echo; echo "--- what Git stores: the small pointer file, not the data ---"
run "cat data.csv.dvc"
run "cat .gitignore                               # DVC told Git to ignore the real CSV"

banner "3. Change the CSV, version 2"
run "printf '6,45,80000,1\n7,23,27000,0\n' >> data.csv"
run "wc -l data.csv"
run "dvc status"
run "dvc add data.csv"
run "git add data.csv.dvc && git commit -q -m 'v2: append 2 rows to data.csv'"
run "git log --oneline"

banner "4. Return to the OLDER version, then forward again"
run "git checkout HEAD~1 -- data.csv.dvc && dvc checkout"
run "wc -l data.csv && cat data.csv               # 6 lines = header + 5 rows = v1"
run "git checkout HEAD -- data.csv.dvc && dvc checkout"
run "wc -l data.csv                               # 8 lines = header + 7 rows = v2 again"
run "git status --short; git log --oneline"

banner "5. Configure remote storage and push"
run "dvc remote add -d storage $REMOTE"
run "git add .dvc/config && git commit -q -m 'Configure DVC remote storage'"
echo "(plain 'dvc push' uploads only the version that is in the workspace, so push v1, then v2)"
run "git checkout HEAD~2 -- data.csv.dvc && dvc checkout && dvc push   # HEAD~2 = the v1 commit (the remote-config commit sits on top), workspace = v1"
run "git checkout HEAD -- data.csv.dvc && dvc checkout && dvc push     # workspace = v2"
run "find $REMOTE -type f                         # two files = two versions of data.csv"

banner "6. Delete the local data (file AND local cache), then pull it back"
run "rm -f data.csv && rm -rf .dvc/cache"
run "ls -A"
run "dvc pull"
run "wc -l data.csv && dvc status"
run "git log --oneline"

banner "7. Bonus: the remote holds BOTH versions (local cache was deleted, so v1 must come from the remote)"
run "git checkout HEAD~2 -- data.csv.dvc && dvc pull && wc -l data.csv   # 6 lines = v1"
run "git checkout HEAD -- data.csv.dvc && dvc checkout && wc -l data.csv  # 8 lines = v2"
run "git status --short                           # clean"
echo; echo "Toy lab finished OK | $(date '+%F %T') | folder: $LAB | remote: $REMOTE"
