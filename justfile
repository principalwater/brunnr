# SPDX-License-Identifier: Apache-2.0

set shell := ["sh", "-eu", "-c"]

default:
    just --list

fmt:
    cargo fmt --all --check

clippy:
    cargo clippy --workspace --all-targets --all-features -- -D warnings

test:
    cargo test --workspace
    cargo test --workspace --all-features

build:
    cargo build --workspace
    cargo build --workspace --all-features

diagrams:
    for source in docs/diagrams/*.mmd; do \
        output="${source%.mmd}.png"; \
        npx -y @mermaid-js/mermaid-cli \
            -i "$source" \
            -o "$output" \
            -c docs/diagrams/mermaid-config.json \
            -b transparent \
            -s 2; \
    done

bench:
    cargo run -p brunnr-bench -- --reps 2

bench-large:
    cargo run -p brunnr-bench -- --reps 2 --seed-corpus benchmarks/large-corpus --results benchmarks/results/large-corpus

bench-xl:
    python3 benchmarks/tools/generate_corpus.py --out xl-corpus --docs 180 --tasks 40
    cargo run -p brunnr-bench -- --reps 2 --seed-corpus benchmarks/xl-corpus --results benchmarks/results/xl-corpus

bench-session:
    python3 benchmarks/tools/generate_corpus.py --out session-corpus --docs 1600 --tasks 60
    cargo run -p brunnr-bench -- --reps 1 --seed-corpus benchmarks/session-corpus --results benchmarks/results/session-corpus

bench-mid:
    python3 benchmarks/tools/generate_corpus.py --out mid-corpus --docs 6400 --tasks 80
    cargo run -p brunnr-bench -- --reps 1 --seed-corpus benchmarks/mid-corpus --results benchmarks/results/mid-corpus

bench-plot:
    python3 benchmarks/tools/plot_scaling.py

bench-check:
    just bench
    just bench-large
    just bench-xl
    just bench-session
    just bench-mid
    git diff --exit-code -- benchmarks/results/sample-run benchmarks/results/large-corpus benchmarks/results/xl-corpus benchmarks/results/session-corpus benchmarks/results/mid-corpus

ci: fmt clippy test build
