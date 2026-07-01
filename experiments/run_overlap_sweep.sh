#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

DOMAINS_A=(physics math biology chemistry)
DOMAINS_B=(biology chemistry physics math)

# Optional extra args forwarded to the analysis script (e.g. --model-slug, --threshold)
EXTRA_ARGS="${@}"

for domain_a in "${DOMAINS_A[@]}"; do
    for domain_b in "${DOMAINS_B[@]}"; do
        [[ "$domain_a" == "$domain_b" ]] && continue
        echo "========================================"
        echo "domain_a=${domain_a}  domain_b=${domain_b}"
        echo "========================================"
        "$PYTHON" "$SCRIPT_DIR/analyze_overlap_firing_rates.py" \
            --domain-a "$domain_a" \
            --domain-b "$domain_b" \
            $EXTRA_ARGS \
            || echo "[WARN] failed for ${domain_a} vs ${domain_b}, continuing"
    done
done

echo "Done. Results written under results/"
