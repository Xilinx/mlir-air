#!/usr/bin/env bash
# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Disk guard for the nightly LLM runner, whose disk is shared by the persistent
# HF/derived-weight caches and each run's build tree.
#
#   preflight   fail unless free space covers the last run's measured growth
#               (never less than MIN_FREE_GB)
#   watch PID   sample free space; below ABORT_FREE_GB, stop PID's descendants
#               so the job fails with its logs instead of losing the runner
#   record      store this run's growth for the next preflight
set -u +e +o pipefail

STATE_DIR="${CI_DISK_STATE:-$HOME/.cache/mlir-air/ci-disk}"
RUN_DIR="${RUNNER_TEMP:-/tmp}/ci-disk"
MIN_FREE_GB="${MIN_FREE_GB:-60}"
MARGIN_PCT="${MARGIN_PCT:-20}"
ABORT_FREE_GB="${ABORT_FREE_GB:-15}"    # room left for the reporting steps
INTERVAL="${INTERVAL:-30}"
GRACE="${GRACE:-30}"                    # seconds between TERM and KILL

used_kb() { df -Pk "$HOME" | awk 'NR==2 {print $3}'; }
free_kb() { df -Pk "$HOME" | awk 'NR==2 {print $4}'; }
gb() { awk -v k="$1" 'BEGIN {printf "%.1f", k / 1048576}'; }

report() {
  df -h "$HOME"
  du -xh --max-depth=2 "$HOME/.cache" "${GITHUB_WORKSPACE:-.}" /tmp 2>/dev/null |
    sort -rh | head -25
}

descendants() {  # descendants PID SKIP: PID's process tree, minus SKIP's subtree
  local c
  for c in $(ps -o pid= --ppid "$1"); do
    [ "$c" = "$2" ] && continue
    echo "$c"
    descendants "$c" "$2"
  done
}

starttime() {  # field 22 of /proc/PID/stat; the comm field may contain spaces
  sed 's/.*) //' "/proc/$1/stat" 2>/dev/null | awk '{print $20}'
}

case "${1:-}" in
preflight)
  mkdir -p "$STATE_DIR" "$RUN_DIR"
  rm -f "$RUN_DIR"/*
  used_kb > "$RUN_DIR/start_used_kb"
  used_kb > "$RUN_DIR/max_used_kb"
  need_gb=$MIN_FREE_GB
  basis="default (no measured run yet)"
  if [ -s "$STATE_DIR/growth_kb" ]; then
    read -r growth note < "$STATE_DIR/growth_kb"
    measured=$(awk -v k="$growth" -v m="$MARGIN_PCT" 'BEGIN {printf "%d", k * (100 + m) / 100 / 1048576 + 1}')
    basis="last run grew $(gb "$growth") GiB${note:+ ($note)} + ${MARGIN_PCT}%, floor ${MIN_FREE_GB} GiB"
    [ "$measured" -gt "$need_gb" ] && need_gb=$measured
  fi
  free=$(free_kb)
  echo "disk preflight: $(gb "$free") GiB free, need ${need_gb} GiB: $basis"
  if [ "$free" -lt $((need_gb * 1048576)) ]; then
    report
    echo "::error::Runner disk has $(gb "$free") GiB free; this run needs ${need_gb} GiB ($basis). Free space on the runner or reduce the model set."
    exit 1
  fi
  ;;
watch)
  pid="$2"
  last_log=0
  touch "$RUN_DIR/watched"
  while kill -0 "$pid" 2>/dev/null; do
    used=$(used_kb)
    free=$(free_kb)
    [ "$used" -gt "$(cat "$RUN_DIR/max_used_kb")" ] && echo "$used" > "$RUN_DIR/max_used_kb"
    if [ "$free" -lt $((ABORT_FREE_GB * 1048576)) ]; then
      touch "$RUN_DIR/aborted"
      # Pin each target by start time: a worker whose parent exits on TERM is
      # reparented out of the tree, and its pid may be reused before the KILL.
      targets=()
      for p in $(descendants "$pid" "$$"); do
        targets+=("$p:$(starttime "$p")")
      done
      kill -TERM "${targets[@]%%:*}" 2>/dev/null
      echo "::error::Runner disk down to $(gb "$free") GiB free (abort threshold ${ABORT_FREE_GB} GiB); stopped the tests."
      report
      sleep "$GRACE"
      for t in "${targets[@]}"; do
        [ "$(starttime "${t%%:*}")" = "${t#*:}" ] && kill -KILL "${t%%:*}" 2>/dev/null
      done
      mapfile -t left < <(descendants "$pid" "$$")
      kill -KILL "${left[@]}" 2>/dev/null
      exit 0
    fi
    if [ $((SECONDS - last_log)) -ge 600 ]; then
      echo "disk: $(gb "$free") GiB free, grown $(gb $((used - $(cat "$RUN_DIR/start_used_kb")))) GiB this run"
      last_log=$SECONDS
    fi
    sleep "$INTERVAL"
  done
  ;;
aborted)
  [ -f "$RUN_DIR/aborted" ]
  exit $?
  ;;
record)
  [ -f "$RUN_DIR/watched" ] || exit 0  # the tests never started
  growth=$(($(cat "$RUN_DIR/max_used_kb") - $(cat "$RUN_DIR/start_used_kb")))
  [ "$growth" -lt 0 ] && growth=0
  note=""
  [ -f "$RUN_DIR/aborted" ] && note="aborted, lower bound"
  mkdir -p "$STATE_DIR"
  echo "$growth $note" > "$STATE_DIR/growth_kb"
  line="Runner disk: grew $(gb "$growth") GiB this run${note:+ ($note)}, $(gb "$(free_kb)") GiB free now"
  echo "$line"
  [ -n "${GITHUB_STEP_SUMMARY:-}" ] && echo "$line" >> "$GITHUB_STEP_SUMMARY"
  ;;
*)
  echo "usage: $0 preflight | watch PID | aborted | record" >&2
  exit 2
  ;;
esac
exit 0
