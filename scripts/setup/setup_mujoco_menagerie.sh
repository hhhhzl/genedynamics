#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODULE_PATH="third_party/mujoco_menagerie"
PATCH_FILE="$ROOT_DIR/third_party/patches/mujoco_menagerie-go2-mjx.patch"
BASE_SHA="a03e87bf13502b0b48ebbf2808928fd96ebf9cf3"
LOCAL_SHA="e5146679f3cfcb327cf759fc9706f5bb0236bd5e"

if [[ $# -gt 1 ]]; then
    echo "Usage: $0 [existing-menagerie-checkout]" >&2
    exit 2
fi

MODULE_DIR="${1:-$ROOT_DIR/$MODULE_PATH}"
if [[ ! -e "$MODULE_DIR/.git" ]]; then
    if [[ $# -ne 0 ]]; then
        echo "Not an initialized MuJoCo Menagerie checkout: $MODULE_DIR" >&2
        exit 1
    fi
    git -C "$ROOT_DIR" submodule update --init --recursive -- "$MODULE_PATH"
fi

MODULE_HEAD="$(git -C "$MODULE_DIR" rev-parse HEAD)"
if [[ "$MODULE_HEAD" != "$BASE_SHA" && "$MODULE_HEAD" != "$LOCAL_SHA" ]]; then
    echo "MuJoCo Menagerie is at $MODULE_HEAD; expected the pinned $BASE_SHA." >&2
    echo "Checkout left unchanged. Preserve your changes before selecting the release asset revision." >&2
    exit 1
fi

if git -C "$MODULE_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "MuJoCo Menagerie Go2 MJX patch is already applied."
elif git -C "$MODULE_DIR" apply --check "$PATCH_FILE"; then
    git -C "$MODULE_DIR" apply "$PATCH_FILE"
    echo "Applied the Generative Dynamics Go2 MJX collision patch."
else
    echo "Go2 MJX patch conflicts with this checkout; no files were changed." >&2
    exit 1
fi
