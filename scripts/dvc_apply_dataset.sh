#!/usr/bin/env bash
# Task 1.4 - APPLY PART: version the real dataset (data/raw) with DVC and store it on DagsHub storage.
# Run from anywhere:  bash scripts/dvc_apply_dataset.sh     (safe to re-run)
# The DagsHub login comes from the git-ignored .env file; the token is never printed or committed.
set -euo pipefail
cd "$HOME/23L0682_A2_Ops"
. .venv/bin/activate
export DVC_NO_ANALYTICS=1
. <(tr -d '\r' < .env)
: "${DAGSHUB_USERNAME:?DAGSHUB_USERNAME missing in .env}" "${DAGSHUB_TOKEN:?DAGSHUB_TOKEN missing in .env}" "${DAGSHUB_REPO:?DAGSHUB_REPO missing in .env}"
REMOTE_URL="https://dagshub.com/${DAGSHUB_USERNAME}/${DAGSHUB_REPO}.dvc"   # DagsHub's DVC storage endpoint for this repo

run() { printf '\n$ %s\n' "$*"; eval "$*"; }
echo "Task 1.4 apply | user: $(whoami) | $(date '+%F %T') | project: 23L0682_A2_Ops"

[ -d .dvc ] || run "dvc init"
run "dvc config core.analytics false          # no anonymous usage reports"

# Earlier I hid data/ from Git with a local-only exclude so the images could not be committed by accident.
# DVC now takes over that job (data/.gitignore), so the exclude must go or Git cannot see data/raw.dvc.
grep -vx 'data/' .git/info/exclude > .git/info/exclude.tmp || true
mv .git/info/exclude.tmp .git/info/exclude

run "dvc add data/raw                         # hash all 975 images, copy them into the DVC cache, write data/raw.dvc"
run "cat data/raw.dvc                         # the small pointer file Git will store"
run "cat data/.gitignore                      # DVC told Git to ignore the real images"

run "dvc remote add -f -d origin $REMOTE_URL"
run "dvc remote modify origin --local auth basic"
run "dvc remote modify origin --local user $DAGSHUB_USERNAME"
printf '\n$ dvc remote modify origin --local password ********    # token read from .env, never printed\n'
dvc remote modify origin --local password "$DAGSHUB_TOKEN"
run "dvc remote list"
run "git check-ignore -v .dvc/config.local    # credentials live in a git-ignored file"

run "dvc push                                 # upload the images to DagsHub storage"
run "dvc status --cloud                       # 'up to date' = remote has everything"
echo; echo "DVC apply finished | $(date '+%F %T')"
