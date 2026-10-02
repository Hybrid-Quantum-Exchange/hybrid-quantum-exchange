# Era One deploy observation — evidence index v0

| Collection metadata | Value | Source pointer |
| --- | --- | --- |
| Topic | Era One deploy observation: Cloudflare Access gate, live verification 2026-10-02 | [Worker 8 task, PR #438](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/438) |
| Collector | Worker 8, Copilot lane only | [PR #438](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/438) |
| Actual collection start | 2026-10-02T07:49:23-05:00 (America/Chicago, CDT) | Local clock reading using `TZ=America/Chicago date`; recorded in this file |
| Method | Read-only GitHub PR bodies, changed-file records, and searches; no live-site or provider access | Collection receipt in `/home/runner/work/hybrid-quantum-exchange/hybrid-quantum-exchange/docs/copilot-waves/wave-1/worker-8/STATUS-WORKER8-v0.md` |

Collection time is not the time of a deployment or live observation. No per-PR collection seconds are assigned.

| Topic claim | Evidence status | Source pointer | Evidence limit |
| --- | --- | --- | --- |
| Era One's deployment is gated by Cloudflare Access. | UNVERIFIED | UNVERIFIED; [PR #438](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/438) names the topic only. | No observation artifact establishing the gate was found in the inspected GitHub records; no gate was accessed or bypassed. |
| Era One's deployment was live-verified on 2026-10-02. | UNVERIFIED | UNVERIFIED; [PR #438](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/438) supplies the requested date only. | No dated live-verification result was found in the inspected records; this worker did not perform live verification. |

| Read-only evidence checked | Source pointer | Result and relevance |
| --- | --- | --- |
| Earlier evidence-index assignment | [PR #224](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/224), changed file `docs/agent-openers/o0008-era-deploy-obs-docs-index.md` | Assignment stub explicitly says “not finished work”; not proof of either topic claim. |
| Earlier claim-verification assignment | [PR #284](https://github.com/Hybrid-Quantum-Exchange/hybrid-quantum-exchange/pull/284), changed file `docs/agent-openers/o0428-era-deploy-obs-verify-claims.md` | Assignment stub, not a verification result; not proof of either topic claim. |
| GitHub code search | UNVERIFIED — no evidence pointer returned by searches scoped to this repository for `"Cloudflare"` and for `"2026-10-02" "Cloudflare"` | No matches returned. Search absence is not proof that the gate or observation does not exist. |

STATUS: Evidence index complete; both deployment-observation claims remain UNVERIFIED. See the separate Worker 8 receipt.
