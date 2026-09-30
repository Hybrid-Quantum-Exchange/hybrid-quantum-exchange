# Vault map (public-safe)

Hybrid-Quantum-Exchange separates **public research** from **private control**.

| Repo | Visibility | What you get here |
| --- | --- | --- |
| This repo (`hybrid-quantum-exchange`) | public | Bottlenecks corpus, Erdős/Aer `LOCAL_SIM` lanes, product shell |
| `oeisdata` (`sequences`) | public | OEIS content fork for sequence lookup |
| `quantum-worker-pilot` | private | Classical hybrid QAOA pilot |
| `quantum-project-ledger` | private | Posture, progress, security rules (no secrets in git) |

Hardware / paid cloud QPU work is gated. Default: **submit locked**. Labels: `LOCAL_SIM` · `CLOUD_SIM` · `REAL_QPU`.

## Operator and automation boundaries

The sole operator face is [DeepNet Chat](https://agenci-main.github.io/deepnet-chat/). This repository's site is a product shell, not an operator console. Do not invent yields or results, or treat simulator output as QPU evidence. QPU status is **Hold**: hardware submission remains locked pending explicit unlock and `REAL_QPU` receipts. Copilot work here is limited to documentation and CI, not QPU or runtime operations.

## CI visibility (2026-09-30)

GitHub Actions lists platform-managed [CodeQL](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/workflows/github-code-scanning/codeql), Dependency Graph, and Copilot cloud-agent workflows. This repository does not contain a tracked `.github/workflows/` build or test gate. A [successful CodeQL run for PR #17](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/actions/runs/36702693795) is evidence for that check on that ref only; it is not a repository-wide test pass, a QPU receipt, or evidence of hardware yield.

This page documents public-facing boundaries only; it does not change runtime behavior or grant permission to submit jobs.
