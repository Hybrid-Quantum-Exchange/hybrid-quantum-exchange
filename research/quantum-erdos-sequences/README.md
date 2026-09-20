# ammo — Erdős quantum sequences

> **Initial port provenance.** HQX PR #1 imported this subtree from
> `AgenCi-MAIN/core-platform-site` PR #179 at
> `58b171fcf9036e89c9a2b066ba2bda212e586b3a`. At HQX main
> `051746a0b08f474e3cc0a6216b89d6424497e4bd`, all 999 lane scripts,
> `run_all.py`, and `requirements.txt` matched that source exactly.
> Unrelated PR #166 history was deliberately excluded.
>
> **Local evidence-runner revision, 2026-09-20.** The runner and documentation
> are now hardened, with focused regression tests and per-run receipts.
> All 999 circuit scripts and requirements remain unchanged. This revision
> does not reproduce the historical full-corpus totals or establish quantum
> advantage, hardware execution, or independent mathematical review.

## Quickstart

```console
pip install -r requirements.txt
python run_all.py --lane 123
python -m unittest -v test_run_all
```

Qiskit and Aer versions are pinned; NumPy has a lower bound. Use an isolated
Python environment. Actual installed versions are recorded in each receipt.
The default without `--lane` executes all 999 scripts; start with a bounded lane.

---
# quantum-erdos-sequences

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

## Run evidence and interpretation

For a bounded local run with dependencies already installed, use:

```console
python run_all.py --lane 123
```

Without `--lane`, the runner requires exactly one source for each problem 1–999
and executes the full corpus. An empty or incomplete corpus, or a missing selected
lane, fails before any lane executes. Dependencies are listed in `requirements.txt`;
Qiskit and Aer are pinned, while NumPy has a lower bound. Receipts record the
installed versions without importing the packages or installing anything.

Every invocation creates a unique `.runs/<UTC timestamp>-<UUID>/` directory,
ignored by Git. It contains a run `manifest.json`, aggregate `RESULTS.json` and
`RESULTS.md` for completed runs, and a subdirectory per attempted lane containing
`receipt.json`, byte-exact `stdout.log`, and byte-exact `stderr.log`. The manifest
records SHA-256 hashes of the runner and requirements, Python and dependency
versions, UTC timestamps, and the selected lanes. Each lane receipt records its
source hash, command, working directory, duration, exit code, log hashes, and errors.
Timeouts, launch failures, missing sources, and corpus-selection errors retain
failure receipts. A filesystem failure that prevents writing the receipt directory
cannot itself be recorded there. Source changes observed during execution invalidate
the result. Hashes identify local bytes; receipts are neither signed nor an
independent audit, and do not capture the complete operating-system environment.

The top-level `RESULTS.json` and `RESULTS.md` remain latest-run convenience copies;
the unique run directories retain earlier evidence. The legacy JSON keys remain:
`ran_ok` means clean execution **and** a valid, unambiguous verdict;
`verified_against_classical` is only a compatibility alias for
`self_reported_demo_pass`. It does **not** establish independent classical
verification or anything about the named Erdos problem. The separate statuses are:

- `execution_status`: clean, failed, timeout, launch/source error, or changed source.
- `protocol_status`: valid, missing verdict, or contradictory verdicts.
- `hypothesis_status`: self-reported pass/fail for the demo, or not established.
- `reviewer_status`: always `not_independently_reviewed`.

Verdict parsing accepts complete, case-sensitive lines, with surrounding whitespace:
bare `PASS`/`FAIL`, optionally followed by a colon or opening parenthesis and an
explanation; `RESULT: PASS/FAIL`; or `PASS/FAIL: PASS/FAIL`. Two explicit legacy
forms are supported: `Overall verdict for Erdos problem #<number>: PASS/FAIL`, and
`Quantum result [<integers>] matches classical answer [<integers>]: PASS` (also
`does NOT match` and `FAIL`). Here `PASS/FAIL` denotes either verdict, except the
literal `PASS/FAIL:` prefix. Narrative occurrences such as `NOTE: PASS/FAIL`,
intermediate check labels, and prose ending in `FAIL` are not verdicts. Both
recognized verdicts in one output, including conflicting words in a verdict's
explanation, fail the protocol check. Repeated identical verdicts are permitted.
Exit 1 plus an unambiguous `FAIL` can be a clean failed experiment; exit 1 plus
`PASS`, other nonzero exits, and Python tracebacks in **either** stream fail
execution. A clean `FAIL` still makes the aggregate command exit nonzero.

Lane scripts use unseeded shots. The runner does not seed simulators or transpilers,
and its metadata is not a seed control. Results can vary; no deterministic rerun,
hardware execution, named-problem proof, or quantum advantage is established.

Run the runner's focused tests without Qiskit or a corpus execution:

```console
python -m unittest -v test_run_all.py
```

## Historical reports

The prior README reported 999 attempted files, 999/999 `ran_ok`, and 993/999 printed
PASS. Those figures predate the fail-closed parser and durable receipts and are
**not revalidated totals**. The old parser could mistake `NOTE: PASS/FAIL` for PASS,
so those aggregates must not be treated as proof. Prior notes named 313, 515, 835,
927 and 935 as threshold-sensitive; that behavior was not rerun for the runner fix.

Problem 906's final printed verdict is FAIL by design.
Along with problems 225, 426, 689, 831, and 910, it is one of the 6 lanes
whose own script says explicitly that no genuine OEIS sequence exists for
that problem and that the substitute circuit run instead verifies nothing
about the named problem; 906 is simply the one of the six that reports that
as an overall FAIL rather than a caveated PASS (a discrepancy in how the
individual lane scripts phrase their own honest-limitation verdict, not a
correctness issue in what each one discloses).

- **Historically reported honest-limitation count** (script's own docstring/notes state that the problem had
  no genuine attached OEIS sequence — `oeis` field is `"N/A"`, `"none"`, or a
  non-numeric placeholder such as `"possible"` — and a substitute finite property was
  used instead): **710**, derived by reading each lane's reported `oeis` field.

## 30 illustrative lanes (historical self-reports)

| # | OEIS | Approach | Prior demo report |
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

**What it is:** a large batch of small quantum circuit demonstrations (almost
all Grover's-algorithm instances on 2–12 qubits) run against the ideal `AerSimulator`,
each reporting its own check against a classical computation of the same
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
  named problem; see "Historical reports" above for which of the six report that as PASS
  (with a caveat) vs. FAIL.
- Other lanes can flip PASS/FAIL between runs due to unseeded shots landing near
  a pass threshold. Shot noise is one possible explanation; the runner does not
  independently diagnose a lane's correctness or stability.
- This is a demonstration/exercise corpus, not a research result: it shows that a
  small Grover search can be built and verified against a classical ground truth for
  many finite combinatorial facts, nothing more.
