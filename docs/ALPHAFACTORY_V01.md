# AlphaFactory V0.1 — Falsification-First Spec

## Goal

Build a Rust research kernel that can reject false alpha ideas before it tries to optimize them.

V0.1 is **not** a live-trading milestone. It is successful only if the research pipeline is point-in-time safe, reproducible, and capable of surviving a truly unseen holdout.

## Non-negotiable invariants

1. **No look-ahead** — a mutation at time `t+k` must never alter any factor value at time `<= t`.
2. **No survivorship-selected universe in historical tests** — the tradable universe must be reconstructed point-in-time.
3. **No train/test reuse** — search/optimization cannot access the final holdout.
4. **No free fills** — later backtests must use next-bar executable prices and explicit costs.
5. **No silent NaN filling** — unavailable history stays unavailable.
6. **Determinism** — identical input data + seed + formula must reproduce identical output.

## V0.1 scope

### Rust core

- postfix formula DSL / bytecode
- deterministic stack VM
- causal delay
- causal rolling mean/std/z-score
- strict warm-up semantics
- invariance tests that mutate future observations

### Next slices

- expression parser + canonical formula hash
- point-in-time dataset schema
- vectorized multi-asset evaluation
- rank IC / ICIR
- turnover and cost model
- walk-forward splitter
- purged/embargoed validation
- formula complexity penalty
- alpha registry

## Frozen research protocol

The initial crypto benchmark will use a deliberately frozen final holdout. During search, the holdout is inaccessible to the optimizer.

Suggested first protocol:

- **Research / train:** 2017-01-01 through 2023-12-31
- **Validation:** 2024-01-01 through 2024-12-31
- **Frozen holdout:** 2025-01-01 through the latest complete 2026 data snapshot

The exact dates may be tightened to match trustworthy point-in-time data availability, but the rule cannot change after the search begins.

## First pass/fail gate

A candidate alpha does **not** pass because it has a high in-sample Sharpe.

At minimum it must:

- be causal by construction and by mutation tests;
- remain directionally consistent on validation;
- retain useful signal on the untouched holdout after realistic costs;
- avoid being a near-duplicate of an already retained alpha;
- survive a subsequent shadow/paper-trading period without parameter changes.

If the system cannot discover at least one candidate that survives this protocol, the correct V0.1 result is **FAIL**, not a relaxed benchmark.

## Relationship to the original AlphaGPT

AlphaGPT remains in the repository as a reference implementation. AlphaFactory keeps the useful idea — symbolic formula generation — but does not trust the original normalization, universe construction, backtest score, or live-risk assumptions without independent reimplementation and validation.

## Compute policy

GitHub is source control only. Research, tests, backtests, and expensive search runs should execute on local/self-hosted compute unless an explicitly free path is chosen.
