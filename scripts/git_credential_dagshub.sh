#!/usr/bin/env bash
# git credential helper for dagshub.com. Reads the login from the git-ignored .env file at run time,
# so the token is never written into .git/config, the repo history or the command line.
# Wired up with:  git config credential.https://dagshub.com.helper '!bash scripts/git_credential_dagshub.sh'
[ "$1" = "get" ] || exit 0
env_file="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
[ -f "$env_file" ] || exit 0
set -a
# tr -d '\r' guards against Windows line endings if .env was saved from Notepad
. <(tr -d '\r' < "$env_file")
set +a
[ -n "$DAGSHUB_TOKEN" ] || exit 0
echo "username=$DAGSHUB_USERNAME"
echo "password=$DAGSHUB_TOKEN"
