#!/bin/sh
# Two-Mac paired serving, supervised: rank 1 first, rank 0 second, both restarted when either stops answering.
# Run it with no verb for the verbs and the environment variables it reads.
set -eu

usage() {
  cat <<'USAGE'
usage: pair_supervise.sh up|down|status|watch

  up      start rank 1 on $PEER, then rank 0 here, and wait for both to answer /health
  down    stop both ranks
  status  pids, ports and each rank's last log lines
  watch   up, then restart both whenever either rank stops answering (unattended)

Nothing here names a machine or a user path: every location is environment overridable.
required  PEER              rank 1's ssh target (key auth: ssh -o BatchMode=yes must work)
          MODEL             the checkpoint both ranks load
          RANK0_SETTINGS    rank 0's --speed-up settings file
          PEER_SETTINGS     rank 1's --speed-up settings file
          BIN               rank 0's tensorfold-native
          PEER_BIN          rank 1's tensorfold-native (same bytes: the check refuses a mismatch)
optional  PORT0=4100 PORT1=4101 CONTEXT=32768 KEEP_WARM=600 SERVE_FLAGS='' SERVE_ENVS=''
          STATE=${TMPDIR:-/tmp}/tensorfold-pair      local logs, pidfile and the port count
          PEER_STATE=${TMPDIR:-/tmp}/tensorfold-pair the same on rank 1's Mac, in that Mac's own TMPDIR
          PEER_MODEL=$MODEL  SSH_OPTS='-o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no'
          POLL=2             seconds between liveness probes and the first restart delay
          STABLE=60          seconds a pair must stay up before the restart delay drops back to POLL
          STOP_WAIT=12       seconds a rank may take to exit after the stop signal, before it is killed
          READY_TIMEOUT=300  seconds either rank may take from process start to answering /health
          ALLOW_MISMATCH=1   start even though the two binaries differ (only for a deliberate A/B test)
USAGE
}


PORT0=${PORT0:-4100}
PORT1=${PORT1:-4101}
CONTEXT=${CONTEXT:-32768}
KEEP_WARM=${KEEP_WARM:-600}
POLL=${POLL:-2}
STABLE=${STABLE:-60}
STOP_WAIT=${STOP_WAIT:-12}
READY_TIMEOUT=${READY_TIMEOUT:-300}
STATE=${STATE:-${TMPDIR:-/tmp}/tensorfold-pair}
PEER_STATE=${PEER_STATE:-${TMPDIR:-/tmp}/tensorfold-pair}
SSH_OPTS=${SSH_OPTS:--o BatchMode=yes -o ConnectTimeout=5 -o StrictHostKeyChecking=no}
SERVE_FLAGS=${SERVE_FLAGS:-}
SERVE_ENVS=${SERVE_ENVS:-}
PEER_MODEL=${PEER_MODEL:-${MODEL:-}}
PEER=${PEER:-}
MODEL=${MODEL:-}
RANK0_SETTINGS=${RANK0_SETTINGS:-}
PEER_SETTINGS=${PEER_SETTINGS:-}
BIN=${BIN:-}
PEER_BIN=${PEER_BIN:-${BIN}}
ALLOW_MISMATCH=${ALLOW_MISMATCH:-0}

LOG0=$STATE/rank0.log
PID0=$STATE/rank0.pid
LOG1=$PEER_STATE/rank1.log
PID1=$PEER_STATE/rank1.pid

say() { printf '[pair] %s\n' "$*"; }

peer() { ssh -n $SSH_OPTS "$PEER" "$@"; }

now() { date +%s; }

# /health is unauthenticated and answers only once the rank is serving, which both ranks reach after the handshake.
health0() { curl -fsS -m 2 -o /dev/null "http://127.0.0.1:$PORT0/health" 2>/dev/null; }
health1() { peer "curl -fsS -m 2 -o /dev/null http://127.0.0.1:$PORT1/health" 2>/dev/null; }

local_pid() { [ -f "$PID0" ] && cat "$PID0" 2>/dev/null || true; }
peer_pid() { peer "cat '$PID1' 2>/dev/null" 2>/dev/null || true; }

alive0() { p=$(local_pid); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
alive1() { peer "kill -0 \$(cat '$PID1' 2>/dev/null) 2>/dev/null" 2>/dev/null; }

need_peer() {
  [ -n "$PEER" ] || { say "set PEER to the rank 1 ssh target"; return 1; }
}

# Only a start needs the checkpoint and the two settings files: stopping a pair must work on a host where the model
# has been moved or was never mounted, which is exactly when someone runs 'down'.
need_up() {
  need_peer || return 1
  [ -n "$MODEL" ] || { say "set MODEL to the checkpoint directory"; return 1; }
  [ -n "$RANK0_SETTINGS" ] || { say "set RANK0_SETTINGS to rank 0's settings file"; return 1; }
  [ -n "$PEER_SETTINGS" ] || { say "set PEER_SETTINGS to rank 1's settings file"; return 1; }
  [ -n "$BIN" ] || { say "set BIN to rank 0's tensorfold-native (PEER_BIN defaults to it)"; return 1; }
  [ -n "$PEER_BIN" ] || { say "set PEER_BIN to rank 1's tensorfold-native"; return 1; }
}

check_binaries() {
  if [ "$ALLOW_MISMATCH" = 1 ]; then say "binary check skipped (ALLOW_MISMATCH=1)"; return 0; fi
  local here there
  here=$(shasum -a 256 "$BIN" | cut -d' ' -f1)
  there=$(peer "shasum -a 256 '$PEER_BIN' | cut -d' ' -f1" 2>/dev/null || echo missing)
  if [ "$here" != "$there" ]; then
    say "the two ranks are not the same build: $BIN $here, $PEER_BIN $there"
    say "fix: copy the same binary to both Macs, or set ALLOW_MISMATCH=1 for a deliberate A/B test"
    return 1
  fi
  say "binaries match: $(printf %s "$here" | cut -c1-12)"
}

# Rank 1 waits for rank 0 inside the handshake, so it cannot answer /health first: start it, let it reach that wait.
start_rank1() {
  peer "mkdir -p '$PEER_STATE'"
  say "starting rank 1 on $PEER (it waits for rank 0 in the pair handshake)"
  peer "cd '$PEER_STATE' && nohup env $SERVE_ENVS '$PEER_BIN' serve '$PEER_MODEL' --speed-up '$PEER_SETTINGS'" \
       "--port $PORT1 --context $CONTEXT --keep-warm $KEEP_WARM --parallel 1 --no-update-check $SERVE_FLAGS" \
       ">$LOG1 2>&1 & echo \$! > '$PID1'"
  sleep 2
  if ! alive1; then
    say "rank 1 exited at once: last lines of $PEER:$LOG1"
    peer "tail -n 5 '$LOG1' 2>/dev/null" || true
    return 1
  fi
  say "rank 1 is up (pid $(peer_pid)), waiting for the handshake"
}

start_rank0() {
  say "starting rank 0 here"
  cd "$STATE"
  # The launcher is a subshell so the rank's death is never this shell's job to report: a supervisor log stays one
  # line per event instead of one per killed child.
  pid=$(
    nohup env $SERVE_ENVS "$BIN" serve "$MODEL" --speed-up "$RANK0_SETTINGS" --port "$PORT0" \
      --context "$CONTEXT" --keep-warm "$KEEP_WARM" --parallel 1 --no-update-check $SERVE_FLAGS >"$LOG0" 2>&1 &
    echo $!
  )
  echo "$pid" >"$PID0"
  say "rank 0 pid $(local_pid)"
}

# Poll until this rank answers /health; the ports come up only after the pair handshake, so the count of refusals
# is the readiness delay, not a startup race the client can lose a request to.
wait_serving() {
  start=$(now)
  refused=0
  while :; do
    if health0 && health1; then
      say "both ranks answer: rank 0 on :$PORT0 and rank 1 on :$PORT1, $(( $(now) - start ))s after rank 0 started"
      [ "$refused" -gt 0 ] && say "rank 0's port refused $refused probes while the pair handshake completed"
      return 0
    fi
    if ! alive0; then
      say "rank 0 exited before it served: last lines of $LOG0"
      tail -n 8 "$LOG0" 2>/dev/null || true
      return 1
    fi
    if ! alive1; then
      say "rank 1 exited before it served: last lines of $PEER:$LOG1"
      peer "tail -n 8 '$LOG1' 2>/dev/null" || true
      return 1
    fi
    refused=$((refused + 1))
    if [ $(( $(now) - start )) -ge "$READY_TIMEOUT" ]; then
      say "no answer after ${READY_TIMEOUT}s: last lines of each log"
      tail -n 8 "$LOG0" 2>/dev/null || true
      peer "tail -n 8 '$LOG1' 2>/dev/null" || true
      return 1
    fi
    sleep 1
  done
}

up() {
  need_up || return 1
  mkdir -p "$STATE"
  command -v curl >/dev/null || { say "curl is required for the readiness probe"; return 1; }
  if health0 && health1; then say "already serving"; return 0; fi
  check_binaries
  down
  start_rank1
  start_rank0
  wait_serving
}

# A stop is TERM, then KILL if the rank is still there: a draining rank exits in a second, and a wedged one must not
# hold a restart. kill() returns 0 for a process that is still shutting down, so each stop waits the pid out first.
stop_local() {
  kill "$1" 2>/dev/null || return 0
  i=0
  while kill -0 "$1" 2>/dev/null && [ "$i" -lt $((STOP_WAIT * 5)) ]; do sleep 0.2; i=$((i + 1)); done
  if kill -0 "$1" 2>/dev/null; then
    say "rank 0 did not stop within ${STOP_WAIT}s: killing it (pid $1)"
    kill -9 "$1" 2>/dev/null || true
    i=0
    while kill -0 "$1" 2>/dev/null && [ "$i" -lt 25 ]; do sleep 0.2; i=$((i + 1)); done
  fi
  say "stopped rank 0 (pid $1)"
}

# The peer runs its own stop loop so the pids, the signals and the waits all happen on that Mac, one line per pid.
stop_peer() {
  pattern=$(basename "${PEER_BIN:-tensorfold-native}")
  peer "for q in \$( { cat '$PID1' 2>/dev/null; pgrep -f '$pattern serve.*--speed-up'; } | sort -u ); do" \
       "kill \$q 2>/dev/null || continue; i=0;" \
       "while kill -0 \$q 2>/dev/null && [ \$i -lt $((STOP_WAIT * 5)) ]; do sleep 0.2; i=\$((i + 1)); done;" \
       "if kill -0 \$q 2>/dev/null; then echo '[pair] rank 1 will not stop: killing it (pid '\$q')';" \
       "kill -9 \$q 2>/dev/null; sleep 1; fi;" \
       "echo '[pair] stopped rank 1 (pid '\$q')';" \
       "done" 2>/dev/null || true
}

down() {
  need_peer || return 1
  p=$(local_pid)
  [ -n "$p" ] && stop_local "$p"
  for q in $(pgrep -f "$(basename "${BIN:-tensorfold-native}") serve.*--speed-up" 2>/dev/null); do
    [ "$q" = "$p" ] && continue
    say "stopping a rank 0 the pidfile missed (pid $q)"
    stop_local "$q"
  done
  stop_peer
  rm -f "$PID0"
  return 0
}

status() {
  need_peer || return 1
  if health0; then say "rank 0: serving on :$PORT0 (pid $(local_pid))"; else say "rank 0: not answering on :$PORT0"; fi
  if health1; then say "rank 1: serving on :$PORT1 (pid $(peer_pid))"; else say "rank 1: not answering on :$PORT1"; fi
  say "last rank 0 lines: $LOG0"
  tail -n 3 "$LOG0" 2>/dev/null || true
  say "last rank 1 lines: $PEER:$LOG1"
  peer "tail -n 3 '$LOG1' 2>/dev/null" || true
}

# One supervisor tick: a rank that stopped answering is stopped on both Macs and started again, so a dead peer
# never leaves a half pair up and no human has to notice. The wait between attempts doubles while the pair keeps
# failing, and drops back to POLL only after a run that stayed up for STABLE seconds: a crash loop must not spin.
watch() {
  trap 'say "supervisor stopping: stopping both ranks"; down; exit 0' INT TERM
  restarts=0
  delay=$POLL
  while :; do
    if up; then
      began=$(now)
      while health0 && health1; do
        sleep "$POLL"
      done
      restarts=$((restarts + 1))
      if ! health0; then say "rank 0 stopped answering: restarting both (restart $restarts)"; fi
      if ! health1; then say "rank 1 stopped answering: restarting both (restart $restarts)"; fi
      if [ $(( $(now) - began )) -ge "$STABLE" ]; then delay=$POLL; fi
    else
      restarts=$((restarts + 1))
      say "the pair did not come up: restarting both (restart $restarts)"
    fi
    down
    say "next attempt in ${delay}s"
    sleep "$delay"
    delay=$((delay * 2))
    [ "$delay" -le 60 ] || delay=60
  done
}

case ${1:-usage} in
  up) up ;;
  down) down ;;
  status) status ;;
  watch) watch ;;
  *) usage; exit 2 ;;
esac
