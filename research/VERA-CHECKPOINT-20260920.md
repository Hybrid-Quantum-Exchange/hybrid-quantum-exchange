# September 20 research checkpoint

This public checkpoint summarizes AI-assisted recovery work. It does not mirror
private worker conversations or operational records. The repository's existing
license is unchanged; no new license selection or T9 release is implied.

## Included work

- Hardened sequence runner, focused regressions, research README and dashboard
  wording from the local recovery candidate.
- Grover known-target study source and arithmetic/mocked-execution tests, made
  portable by replacing the local historical-source default with a relative path.
- Selected unchanged circuit exports and raw arm results from two historical
  matched-seed ideal simulator runs. See `grover-verification/evidence/README.md`
  for the omitted receipts and changed-source boundary.
- This sanitized checkpoint and a SHA-256 inventory of the added study files.

## Findings and remaining work

1. Quantum oracle self-consistency cannot establish that stored sequence answers
   are correct. Require independent classical generators and wrong-data controls.
   Only three independent prefixes were checked in the separate recovered dataset;
   full dataset correctness, missing rebuild inputs and third-party rights remain
   unresolved. That dataset is excluded here.
2. A separate cross-audit candidate clarifies the operational six-state
   teleportation statistic and its 0.70 threshold, and limits each call to 4,000
   aggregate circuit-shots. Its source comes from a separate project and remains
   excluded pending provenance/rights and integration review. Offline fake-backend
   tests do not verify hardware execution.
3. One generic worker pipeline attachment is not locally recovered or tested.
   It is excluded. Chat delivery is not verified artifact delivery.
4. Historical archive cleanup does not prove jobs completed. Private chat records,
   account identifiers, operational logs, machine paths, credentials and legacy
   business material are excluded from this public branch.
5. Rook page access remains unverified; no Rook content or claims are included.

## Next gates

Expand classical controls; recover missing inputs with provenance; prepare a
frozen simulator protocol with independent seeds and stopping rules; retain raw
results and failures; obtain independent review and human acceptance. Review
rights and reproduction coverage before any scientific release. This branch is
for code review, not evidence of deployment, QPU use, quantum advantage or T9
maturity. No real-money activity is involved.

The private 94-file recovery manifest passed a complete hash comparison at this
checkpoint. That integrity check does not validate scientific claims or make the
private bundle a public allowlist. Original private files remain unchanged.
