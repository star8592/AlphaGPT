# AlphaFactory

Clean-room Rust research infrastructure for systematic alpha discovery.

This codebase is developed independently from first principles. It does **not** reuse the original AlphaGPT implementation, model, data pipeline, Solana execution code, or project structure.

## V0.1 goal

Prove or falsify one claim:

> A small research system can discover at least one factor that survives a completely sealed out-of-sample holdout after realistic costs, without look-ahead or survivorship leakage.

## Non-negotiable invariants

- Point-in-time data only.
- Future observations must never change past factor values.
- No survivorship-selected historical universe.
- Search never sees the final holdout.
- Warm-up history stays missing; it is never backfilled from the future.
- Transaction costs and execution assumptions are explicit.
- Same data + formula + seed must be deterministic.
- A failed holdout is a failed experiment; thresholds are not relaxed after seeing it.

## Architecture

```text
Point-in-time data contract
        ↓
Causal feature operators
        ↓
Formula DSL + canonical representation
        ↓
Rust evaluator
        ↓
Walk-forward / purge / embargo
        ↓
IC / ICIR / turnover / drawdown / costs
        ↓
Alpha registry
        ↓
Searchers (later: random / GP / RL / LLM)
```

## Current scope

V0.1 intentionally excludes live trading, exchange credentials, private keys, LLM agents, and neural-network search. Correctness and falsifiability come first.

## Local qualification

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
```

Research compute is intended to run locally/self-hosted. GitHub is source control, not the compute plane.
