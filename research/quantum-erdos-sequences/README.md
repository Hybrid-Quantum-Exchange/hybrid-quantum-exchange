# ammo — Erdős quantum sequences

> **Provenance.** This tree is a byte-identical port of the `research/quantum-erdos-sequences/`
> subtree from branch `claude/erdos-quantum-sequences` of `AgenCi-MAIN/core-platform-site`
> (PR #179). All 999 lane scripts, `run_all.py`, and `requirements.txt` are preserved
> exactly as they were there. The unrelated PR #166 history that contaminated the
> source branch was deliberately left behind — this branch carries only the
> Erdős content. It is the first pull request on the Hybrid Quantum Exchange
> repository.

## Quickstart

```bash
pip install -r requirements.txt   # qiskit, qiskit-aer, numpy (pinned)
python3 run_all.py                # runs all 999 lanes, writes RESULTS.json / RESULTS.md
python3 run_all.py --lane 123     # run a single lane
```

Simulators only — every circuit runs on the ideal Qiskit `AerSimulator`.
No hardware, no network, no credentials needed.

---

## Upstream documentation

This directory holds 999 small, per-problem Python scripts, one for each numbered
problem in the [erdosproblems.com dataset](https://github.com/manman4/erdosproblems)
(`data/problems.yaml`). Each script builds a small quantum circuit — almost always a
Grover search — over a *finite, classically-decidable property* tied to that problem's
associated OEIS sequence, runs it on the ideal Qiskit `AerSimulator` (no noise model,
no real hardware), and checks the measured/amplified output against a classical
ground truth computed from scratch in the same script. Where a problem had no real
attached OEIS sequence (a very common case — many entries in the source data carry
`oeis: ["N/A"]` or a non-numeric placeholder like `"possible"`), the script says so
explicitly in its docstring and substitutes the closest honest, finite, self-computed
property drawn from the problem's own tags, rather than fabricating a sequence tie-in.

## Totals

These are reproducible: run `python3 run_all.py` from this directory (see
`requirements.txt` for pinned dependency versions) and read `RESULTS.json` /
`RESULTS.md`, which that script regenerates from a fresh subprocess run of
every lane and its own printed PASS/FAIL verdict — not a hand-maintained
claim. The figures below are from one such run.

- **Attempted:** 999
- **Produced a file:** 999
- **`ran_ok` (script executed cleanly and printed a verdict):** 999 / 999
- **`verified_against_classical` (lane printed PASS):** 993 / 999

Lane scripts do not seed their shot counts, so a handful of lanes whose
measured probability sits close to their pass threshold can flip between
PASS and FAIL from one run to the next — rerunning `run_all.py` a few times
during this PR's preparation showed problems 313, 515, 835, 927 and 935 in
that category (their ideal, noiseless success probability is real and well
above the uniform baseline, but shot noise near the threshold occasionally
tips them under it). The large majority of lanes are not close to any
threshold and are stable across reruns.

Problem 906 is different: it is not flaky, it fails by design every run.
Along with problems 225, 426, 689, 831, and 910, it is one of the 6 lanes
whose own script says explicitly that no genuine OEIS sequence exists for
that problem and that the substitute circuit run instead verifies nothing
about the named problem; 906 is simply the one of the six that reports that
as an overall FAIL rather than a caveated PASS (a discrepancy in how the
individual lane scripts phrase their own honest-limitation verdict, not a
correctness issue in what each one discloses).

- **Honest-limitation count** (script's own docstring/notes state that the problem had
  no genuine attached OEIS sequence — `oeis` field is `"N/A"`, `"none"`, or a
  non-numeric placeholder such as `"possible"` — and a substitute finite property was
  used instead): **710**, derived by reading each lane's reported `oeis` field.

## 30 of the more interesting / cleanly successful lanes

| # | OEIS | Approach | Verified |
|---|------|----------|----------|
| 1 | A276661 | Grover search over 4-qubit membership strings finds a 3-subset of {1..4} with all-distinct subset sums | PASS |
| 3 | A003002 | Grover search over 2-colorings of {1..8} for AP-3-free colorings, witnessing van der Waerden W(2,3)=9 | PASS |
| 9 | A006286 | Grover search for k with N−2^k prime (de Polignac-style), witnesses matched to classical search | PASS |
| 14 | A143824 | Grover search over 64 subsets of {1..6} for maximum Sidon sets, B(6)=4 | PASS |
| 30 | A003022 | Grover search over 4-subsets of {0..6} for Sidon witnesses certifying A003022(4)=6 | PASS |
| 52 | A263996 | Grover search over 3-subsets of {1..8} minimizing \|sumset ∪ productset\|, reproduces a(3)=7 | PASS |
| 60 | A006855 | Grover search over K4 edge-subsets for maximum C4-free graphs, a(4)=4 | PASS |
| 67 | A181740, A237695 | Grover search over 64 length-6 ±1 sequences for Erdos-discrepancy-≤1 witnesses | PASS |
| 76 | A060407 | Grover search over K4 2-colorings for a monochromatic-triangle-free coloring (R(3,3)=6) | PASS |
| 77 | A059442 | Grover search over K5 edge-colorings witnessing R(3,3)>5 | PASS |
| 89 | A186704 | Grover search over 4-point configurations minimizing distinct pairwise distances | PASS |
| 104 | A003829 | Grover search over point triples for circumradius exactly 1 | PASS |
| 131 | A068063 | Grover search over subsets of {1..6} for maximum nondividing subset, a(6)=2 | PASS |
| 140 | A003002 | Grover search over [0,64) marking the base-3-no-digit-2 (Stanley sequence) construction | PASS |
| 144 | A005279 | Grover search over [0,63] for the unique primitive abundant number, n=20 | PASS |
| 193 | A231255 | Grover search over length-3 lattice walks avoiding 3 collinear points | PASS |
| 219 | A005115, A113827, A123556 | Grover search over 64 (a,d) pairs for 3-term prime arithmetic progressions (Green-Tao) | PASS |
| 270 | A073016 | Grover search recovering digit-9 positions in the decimal expansion of Σ1/C(2n,n) | PASS |
| 323 | A004831 | Grover search over (a,b) pairs for a^4+b^4=17 | PASS |
| 340 | A005282 | Grover search finding the next term of the greedy Mian-Chowla Sidon sequence | PASS |
| 359 | A002048 | Grover search over subset sums of the first 7 segmented (power-of-2) numbers | PASS |
| 373 | A003135 | Grover search verifying the factorial identity 10! = 6!·7! | PASS |
| 408 | A049108 | Grover search over n=1..16 for iterated-Euler-phi step counts equal to a target | PASS |
| 449 | A399440 | Grover search over divisor-index pairs of n=12 counting the A399440 witnesses | PASS |
| 528 | A387897, A156816 | Grover search over self-avoiding lattice walks of length 4 | PASS |
| 730 | A129515 | Reversible ripple-carry adder circuit computing a Kummer's-theorem carry count | PASS |
| 798 | A116446 | Grover search recovering the minimal point-line covering set for a 4-point grid | PASS |
| 829 | A025455, A025468 | Grover search over cube pairs finding the two taxicab-1729 representations | PASS |
| 850 | A343101 | Grover search over [1,63] for the Erdos-Woods radical-equality witness at offset 16 | PASS |
| 993 | A000055 | Grover search over K4 edge-subsets amplifying labeled spanning trees (Cayley's formula) | PASS |

## What this library does and does not establish

**What it is:** a large batch of small, individually-verified quantum circuits (almost
all Grover's-algorithm instances on 2–12 qubits) run against the ideal `AerSimulator`,
each cross-checked in the same script against a classical computation of the same
finite property. The problems and their OEIS pointers are sourced from
[github.com/manman4/erdosproblems](https://github.com/manman4/erdosproblems); no
problem's actual open mathematical content is resolved by anything here.

**What it does not establish:**
- No claim of quantum advantage. Every instance searches a space of at most a few
  thousand basis states — trivial classically, and the classical answer is computed
  first in every script and used to build or check the oracle.
- No real quantum hardware was used. Every circuit ran on the noiseless, ideal
  `AerSimulator`; none of this says anything about behavior on physical qubits, under
  noise, or at any scale beyond a handful of qubits.
- No Erdos problem is solved, disproved, or otherwise advanced here. Most of the 999
  problems are open, and most of the 999 problems in this dataset have no OEIS
  sequence at all (710 of the lanes say so explicitly and substitute a smaller, honest,
  self-computed finite property in the same subject area, clearly documented as a
  limitation in that script's own docstring).
- The 6 honest-limitation-by-design lanes (225, 426, 689, 831, 906, 910) are
  explicit about testing an unrelated placeholder circuit, not anything about the
  named problem; see "Totals" above for which of the six report that as PASS
  (with a caveat) vs. FAIL.
- A handful of other lanes (see "Totals" above) can flip PASS/FAIL between runs
  due to unseeded shots landing near a pass threshold; this is normal
  binomial variance, not a correctness bug, and is why totals here are
  described as reproducible via `run_all.py` rather than as a fixed number.
- This is a demonstration/exercise corpus, not a research result: it shows that a
  small Grover search can be built and verified against a classical ground truth for
  many finite combinatorial facts, nothing more.
