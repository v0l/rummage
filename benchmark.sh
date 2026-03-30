#!/usr/bin/env bash
# =============================================================================
# benchmark.sh - Rummage PoW performance benchmark suite
#
# Builds the Rust CLI with different -D flags passed via POW_DEFINES env var,
# then benchmarks each variant across difficulty levels and event types
# (including --fast mode).
#
# Usage:
#   ./benchmark.sh                    # Run all variants (full suite)
#   ./benchmark.sh --quick            # Quick run (fewer variants, shorter duration)
#   ./benchmark.sh --variant <name>   # Run a specific variant only
#   ./benchmark.sh --list             # List all variant names
#   ./benchmark.sh --duration <sec>   # Set measurement duration (default: 15)
#   ./benchmark.sh --difficulties "32 34 36"  # Custom difficulty levels
#   ./benchmark.sh --skip-build       # Reuse last built binary (runtime-only tests)
#
# Results are written to benchmark_results/benchmark_<timestamp>.csv
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_DIR="$SCRIPT_DIR/rust"
BINARY="$RUST_DIR/target/release/rummage"
RESULTS_DIR="$SCRIPT_DIR/benchmark_results"

# ---- Defaults ----

DURATION=15
WARMUP=5
DIFFICULTIES=(32 34 36)
QUICK_MODE=0
SKIP_BUILD=0
SPECIFIC_VARIANT=""

# Test event (same as PERFORMANCE.md)
TEST_EVENT='{"id":"","pubkey":"79c2cae114ea28a681e1ba5ebc76007ed87f86694f35f782483f4e4c2d45b96f","created_at":1234567890,"kind":1,"tags":[["e","abc123"],["p","def456"]],"content":"hello world","sig":""}'

# Long suffix event (triggers constant tail block W schedules)
LONG_EVENT='{"id":"","pubkey":"79c2cae114ea28a681e1ba5ebc76007ed87f86694f35f782483f4e4c2d45b96f","created_at":1234567890,"kind":1,"tags":[["e","abc123"],["p","def456"],["t","benchmark"],["t","performance"]],"content":"This is a longer content string designed to produce a large suffix that triggers constant tail block W schedule pre-computation for benchmarking purposes.","sig":""}'

# ---- Parse arguments ----

while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            QUICK_MODE=1; DURATION=10; WARMUP=3; shift ;;
        --duration)
            DURATION="$2"; shift 2 ;;
        --difficulties)
            IFS=' ' read -ra DIFFICULTIES <<< "$2"; shift 2 ;;
        --variant)
            SPECIFIC_VARIANT="$2"; shift 2 ;;
        --skip-build)
            SKIP_BUILD=1; shift ;;
        --list)
            cat <<'EOF'
Available variants:
  baseline              Default (NONCES=256, variable nonce, TPB=256, 2 streams)
  nonces_128            NONCES_PER_THREAD=128
  nonces_512            NONCES_PER_THREAD=512
  fixed_nonce           USE_FIXED_WIDTH_NONCE=1
  tpb_128               POW_THREADS_PER_BLOCK=128
  tpb_512               POW_THREADS_PER_BLOCK=512
  streams_1             NUM_STREAMS=1
  streams_4             NUM_STREAMS=4
  nonces128_fixed       NONCES=128 + fixed-width nonce
  minimal               NONCES=128, 1 stream
  maximal               NONCES=512, 4 streams, TPB=512

Each variant is tested with standard, --fast, and long-suffix events.
EOF
            exit 0 ;;
        --help|-h)
            sed -n '2,18p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Variant definitions ----
# FORMAT: NAME|POW_DEFINES_STRING

declare -a VARIANTS=(
    "baseline|"
    "nonces_128|-DNONCES_PER_THREAD=128"
    "nonces_512|-DNONCES_PER_THREAD=512"
    "fixed_nonce|-DUSE_FIXED_WIDTH_NONCE=1"
    "tpb_128|-DPOW_THREADS_PER_BLOCK=128"
    "tpb_512|-DPOW_THREADS_PER_BLOCK=512"
    "streams_1|-DNUM_STREAMS=1"
    "streams_4|-DNUM_STREAMS=4"
    "nonces128_fixed|-DNONCES_PER_THREAD=128 -DUSE_FIXED_WIDTH_NONCE=1"
    "minimal|-DNONCES_PER_THREAD=128 -DNUM_STREAMS=1"
    "maximal|-DNONCES_PER_THREAD=512 -DPOW_THREADS_PER_BLOCK=512 -DNUM_STREAMS=4"
)

if [[ "$QUICK_MODE" == "1" ]]; then
    VARIANTS=(
        "baseline|"
        "fixed_nonce|-DUSE_FIXED_WIDTH_NONCE=1"
        "nonces_128|-DNONCES_PER_THREAD=128"
        "streams_4|-DNUM_STREAMS=4"
    )
    DIFFICULTIES=(32 36)
fi

if [[ -n "$SPECIFIC_VARIANT" ]]; then
    FILTERED=()
    for v in "${VARIANTS[@]}"; do
        IFS='|' read -r name _ <<< "$v"
        [[ "$name" == "$SPECIFIC_VARIANT" ]] && FILTERED+=("$v")
    done
    if [[ ${#FILTERED[@]} -eq 0 ]]; then
        echo "Error: Unknown variant '$SPECIFIC_VARIANT'. Use --list to see variants."
        exit 1
    fi
    VARIANTS=("${FILTERED[@]}")
fi

# ---- Helpers ----

if [[ -t 1 ]]; then
    BOLD="\033[1m" GREEN="\033[32m" YELLOW="\033[33m"
    CYAN="\033[36m" RED="\033[31m" RESET="\033[0m"
else
    BOLD="" GREEN="" YELLOW="" CYAN="" RED="" RESET=""
fi

log()  { echo -e "${BOLD}[bench]${RESET} $*"; }
ok()   { echo -e "${GREEN}[  ok ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[warn ]${RESET} $*"; }
err()  { echo -e "${RED}[error]${RESET} $*" >&2; }

build_variant() {
    local defines="$1"
    log "Building: POW_DEFINES=\"${defines}\" cargo build --release"
    # Force rummage-sys to recompile by removing its build artifacts
    rm -rf "$RUST_DIR/target/release/build/rummage-sys-"* 2>/dev/null || true
    if ! POW_DEFINES="$defines" cargo build --release --manifest-path "$RUST_DIR/Cargo.toml" 2>&1 | tail -5; then
        err "Build failed"; return 1
    fi
    [[ -x "$BINARY" ]] || { err "Binary not found: $BINARY"; return 1; }
    ok "Build successful"
}

# Run the miner for a fixed duration, return the MH/s rate
# Args: difficulty event_json label [extra_flags...]
run_single_benchmark() {
    local difficulty="$1" event_json="$2" label="$3"
    shift 3
    local extra_flags=("$@")
    local total_time=$((WARMUP + DURATION))
    local logfile; logfile=$(mktemp)

    # Use high difficulty so it never finds a solution during measurement
    local bench_diff=$difficulty
    (( bench_diff < 48 )) && bench_diff=48

    "$BINARY" pow --event "$event_json" --difficulty "$bench_diff" \
        "${extra_flags[@]}" > "$logfile" 2>&1 &
    local pid=$!

    sleep "$total_time" 2>/dev/null || true

    kill -INT "$pid" 2>/dev/null || true
    local waited=0
    while kill -0 "$pid" 2>/dev/null && (( waited < 5 )); do
        sleep 1; (( waited++ )) || true
    done
    kill -9 "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true

    # Extract MH/s from progress lines
    local rate
    rate=$(grep -oP '[\d.]+\s*MH/s' "$logfile" | tail -1 | grep -oP '[\d.]+' || true)
    [[ -z "$rate" ]] && rate=$(grep -oP 'Rate:\s+[\d.]+' "$logfile" | tail -1 | grep -oP '[\d.]+' || true)

    if [[ -z "$rate" ]]; then
        warn "No MH/s found for $label @ ${difficulty}-bit"
        head -20 "$logfile"
        rate="FAIL"
    fi

    # Grab GPU name once
    if [[ -z "${GPU_NAME:-}" ]]; then
        GPU_NAME=$(grep -oP 'PoW Miner GPU: \K[^(]+' "$logfile" | head -1 | xargs || echo "unknown")
    fi

    rm -f "$logfile"
    echo "$rate"
}

# ---- Main ----

log ""
log "======================================================================"
log "  Rummage PoW Benchmark Suite"
log "======================================================================"
log ""
log "  Binary:       $BINARY (Rust CLI)"
log "  Warmup:       ${WARMUP}s"
log "  Measurement:  ${DURATION}s"
log "  Difficulties: ${DIFFICULTIES[*]}"
log "  Variants:     ${#VARIANTS[@]}"
log "  Quick mode:   $([ "$QUICK_MODE" = "1" ] && echo "yes" || echo "no")"
log ""

mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CSV_FILE="$RESULTS_DIR/benchmark_${TIMESTAMP}.csv"
GPU_NAME=""

echo "variant,pow_defines,event_type,difficulty,mh_per_sec" > "$CSV_FILE"

declare -a ALL_RESULTS=()

# Event configurations: name, json, label, extra CLI flags
# "standard" = normal nonce-in-tag mode
# "fast"     = --fast mode (nonce appended to content, ~2B suffix)
# "long"     = long suffix (triggers constant tail block W schedules)
EVENT_TYPES=("standard" "fast" "long")
EVENT_JSONS=("$TEST_EVENT" "$TEST_EVENT" "$LONG_EVENT")
EVENT_LABELS=("standard (nonce tag)" "fast (nonce in content)" "long (const tail blocks)")
EVENT_FLAGS=("" "--fast" "")

if [[ "$QUICK_MODE" == "1" ]]; then
    EVENT_TYPES=("standard" "fast")
    EVENT_JSONS=("$TEST_EVENT" "$TEST_EVENT")
    EVENT_LABELS=("standard (nonce tag)" "fast (nonce in content)")
    EVENT_FLAGS=("" "--fast")
fi

TOTAL_RUNS=$(( ${#VARIANTS[@]} * ${#EVENT_TYPES[@]} * ${#DIFFICULTIES[@]} ))
RUN=0

for variant_def in "${VARIANTS[@]}"; do
    IFS='|' read -r vname vdefines <<< "$variant_def"

    log ""
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "  Variant: ${CYAN}${vname}${RESET}"
    log "  Defines: ${vdefines:-(defaults)}"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ "$SKIP_BUILD" != "1" ]]; then
        if ! build_variant "$vdefines"; then
            err "Skipping variant $vname"; continue
        fi
    fi

    for ei in "${!EVENT_TYPES[@]}"; do
        etype="${EVENT_TYPES[$ei]}"
        ejson="${EVENT_JSONS[$ei]}"
        elabel="${EVENT_LABELS[$ei]}"
        eflags="${EVENT_FLAGS[$ei]}"

        for diff in "${DIFFICULTIES[@]}"; do
            (( RUN++ )) || true
            log "  [${RUN}/${TOTAL_RUNS}] ${etype} @ ${diff}-bit..."

            if [[ -n "$eflags" ]]; then
                rate=$(run_single_benchmark "$diff" "$ejson" "$elabel" $eflags)
            else
                rate=$(run_single_benchmark "$diff" "$ejson" "$elabel")
            fi

            if [[ "$rate" == "FAIL" ]]; then
                warn "  FAILED"
            else
                ok "  ${rate} MH/s"
            fi

            echo "${vname},${vdefines},${etype},${diff},${rate}" >> "$CSV_FILE"
            ALL_RESULTS+=("${vname}|${etype}|${diff}|${rate}")
        done
    done
done

# ---- Summary tables ----

log ""
log "======================================================================"
log "  RESULTS"
log "======================================================================"
[[ -n "$GPU_NAME" ]] && log "  GPU: $GPU_NAME"
log "  ${DURATION}s measurement + ${WARMUP}s warmup per run"
log ""

# Print a table for a given event type
print_table() {
    local filter_event="$1" title="$2"

    log "$title"
    log ""
    printf "${BOLD}%-22s" "Variant"
    for diff in "${DIFFICULTIES[@]}"; do
        printf " %12s" "${diff}-bit"
    done
    printf "${RESET}\n"
    printf '%*s\n' "$((22 + ${#DIFFICULTIES[@]} * 13))" '' | tr ' ' '─'

    for variant_def in "${VARIANTS[@]}"; do
        IFS='|' read -r vname _ <<< "$variant_def"
        printf "%-22s" "$vname"
        for diff in "${DIFFICULTIES[@]}"; do
            found=""
            for r in "${ALL_RESULTS[@]}"; do
                IFS='|' read -r rv re rd rr <<< "$r"
                if [[ "$rv" == "$vname" && "$re" == "$filter_event" && "$rd" == "$diff" ]]; then
                    found="$rr"; break
                fi
            done
            if [[ -z "$found" || "$found" == "FAIL" ]]; then
                printf " %12s" "FAIL"
            else
                printf " %9s MH/s" "$found"
            fi
        done
        printf "\n"
    done
    log ""
}

print_table "standard" "Standard mode (nonce in tag):"
print_table "fast" "Fast mode (nonce in content, --fast):"
print_table "long" "Long suffix (constant tail blocks):"

# Relative performance vs baseline
log "Relative to baseline (standard event):"
log ""

declare -A BASELINE
for r in "${ALL_RESULTS[@]}"; do
    IFS='|' read -r rv re rd rr <<< "$r"
    [[ "$rv" == "baseline" && "$re" == "standard" ]] && BASELINE[$rd]="$rr"
done

printf "${BOLD}%-22s" "Variant"
for diff in "${DIFFICULTIES[@]}"; do
    printf " %12s" "${diff}-bit"
done
printf "${RESET}\n"
printf '%*s\n' "$((22 + ${#DIFFICULTIES[@]} * 13))" '' | tr ' ' '─'

for variant_def in "${VARIANTS[@]}"; do
    IFS='|' read -r vname _ <<< "$variant_def"
    printf "%-22s" "$vname"
    for diff in "${DIFFICULTIES[@]}"; do
        found=""
        for r in "${ALL_RESULTS[@]}"; do
            IFS='|' read -r rv re rd rr <<< "$r"
            if [[ "$rv" == "$vname" && "$re" == "standard" && "$rd" == "$diff" ]]; then
                found="$rr"; break
            fi
        done
        base="${BASELINE[$diff]:-}"
        if [[ -z "$found" || "$found" == "FAIL" || -z "$base" || "$base" == "FAIL" ]]; then
            printf " %12s" "N/A"
        else
            pct=$(awk "BEGIN { printf \"%.1f\", (($found - $base) / $base) * 100 }")
            if awk "BEGIN { exit !($pct >= 0) }"; then
                printf " ${GREEN}%+9.1f%%${RESET}  " "$pct" 2>/dev/null || printf " %+11s%%" "$pct"
            else
                printf " ${RED}%+9.1f%%${RESET}  " "$pct" 2>/dev/null || printf " %+11s%%" "$pct"
            fi
        fi
    done
    printf "\n"
done

# Fast mode speedup vs standard (same variant)
log ""
log "Fast mode speedup vs standard (same variant):"
log ""

printf "${BOLD}%-22s" "Variant"
for diff in "${DIFFICULTIES[@]}"; do
    printf " %12s" "${diff}-bit"
done
printf "${RESET}\n"
printf '%*s\n' "$((22 + ${#DIFFICULTIES[@]} * 13))" '' | tr ' ' '─'

for variant_def in "${VARIANTS[@]}"; do
    IFS='|' read -r vname _ <<< "$variant_def"
    printf "%-22s" "$vname"
    for diff in "${DIFFICULTIES[@]}"; do
        std_rate="" fast_rate=""
        for r in "${ALL_RESULTS[@]}"; do
            IFS='|' read -r rv re rd rr <<< "$r"
            if [[ "$rv" == "$vname" && "$rd" == "$diff" ]]; then
                [[ "$re" == "standard" ]] && std_rate="$rr"
                [[ "$re" == "fast" ]] && fast_rate="$rr"
            fi
        done
        if [[ -z "$std_rate" || "$std_rate" == "FAIL" || -z "$fast_rate" || "$fast_rate" == "FAIL" ]]; then
            printf " %12s" "N/A"
        else
            pct=$(awk "BEGIN { printf \"%.1f\", (($fast_rate - $std_rate) / $std_rate) * 100 }")
            if awk "BEGIN { exit !($pct >= 0) }"; then
                printf " ${GREEN}%+9.1f%%${RESET}  " "$pct" 2>/dev/null || printf " %+11s%%" "$pct"
            else
                printf " ${RED}%+9.1f%%${RESET}  " "$pct" 2>/dev/null || printf " %+11s%%" "$pct"
            fi
        fi
    done
    printf "\n"
done

log ""
log "CSV saved to: ${CSV_FILE}"
log "Benchmark complete!"
