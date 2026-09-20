# Grover 1-of-16 known-target demonstrator

This separate recovery study provides a reproducible experiment for the toy
Grover circuit associated with the historical lane-123 script. It cannot
recover that unseeded run's missing evidence. It is **not an Erdős-123 proof,
problem-specific verification, quantum advantage, or speedup demonstration**.
The original `problem_123.py` is read only for its SHA-256 provenance and is
never modified, imported, or executed by this study.

The frozen protocol uses four qubits, known target 13 (`1101`), three Grover
iterations, and 4,096 shots per arm. Simulator and transpiler seeds are 1729.
Two controls use the same sample count: uniform classical sampling (Python
`random.Random(1729)`) and the same Grover construction marking 12 (`1100`).
The classical reference is intentionally a uniform reference, not the best
classical method for a known answer. No primality test occurs in the circuit.

For a single marked state, the independent analytic prediction is
`p = sin²(7 asin(1/4)) = 63001/65536`; each of the other 15 states has
probability `169/65536`. The manifest records all 16 values for each arm.
Source, transpiled, and QASM-round-trip statevector probabilities must match
the relevant analytic distribution within `1e-12` absolute error.

Sampling uses a predeclared simultaneous Hoeffding bound:
`epsilon = sqrt(log(2*48/0.01)/(2*4096))`. All 48 observed bins must be within
epsilon of their declared probabilities. The positive arm must have mode 13,
the wrong-oracle arm mode 12, and target-13 frequency in the positive arm must
exceed 0.5 and each control's target-13 frequency by more than `2*epsilon`.
No threshold is tuned after seeing a run. A statistical failure is retained.

## Review and execution

The root agent executes the actual study only after independent code/protocol
review. These commands use the existing isolated Python environment; no package
installation is performed by this code.

```powershell
$python = 'python'
New-Item -ItemType Directory -Path protocols -Force | Out-Null
& $python -B -m unittest discover -s tests -v
& $python -B study.py prepare --output protocols\grover-public-v1.json
# Review the frozen manifest, its printed SHA-256, and the source files first.
& $python -B study.py execute --manifest protocols\grover-public-v1.json --expected-manifest-sha256 '<reviewed SHA-256>' --output-root runs
```

`prepare` performs no experiment. It creates a new manifest and sidecar hash
exclusively, including hashes of this study's source, the original lane source,
and installed Qiskit/Aer/NumPy distribution files. Remaining installed packages
have version/RECORD metadata fingerprints, not full content attestation.
This is local environment provenance, not independent supply-chain validation.
Frozen study source, original source, and dependencies are checked before and
after execution. A mismatch stops acceptance; a new reviewed protocol is needed
for a changed method. Never edit a frozen manifest to conceal a failed run.

Each execution creates a unique directory with a start receipt, exact frozen
manifest copy, three per-arm JSON artifacts with counts and timestamps, two
circuit text exports, two OpenQASM 2.0 files, and a final receipt with artifact
hashes and elapsed time. The receipt separates `execution_status`,
`protocol_status`, and `hypothesis_status`. Exit 0 requires `complete`, `passes`,
and `passes` respectively. Independent review remains `pending`; automated
checks never set human acceptance. Failed and nonpassing runs are preserved.

## Circuit semantics and limits

Qubit `i` encodes integer bit `i`; count labels are ordered `q3 q2 q1 q0`.
The oracle phase-flips exactly the marked basis state. The diffuser is
`H^4 X^4 MCZ X^4 H^4`, differing from the conventional reflection only by a
global minus sign. There are exactly three oracle/diffuser pairs. The measured
circuit is transpiled into `u3` and `cx`, with optimization level 1 and seed
1729. Aer uses noiseless statevector simulation and one CPU thread.

OpenQASM 2.0 export is checked with Qiskit's strict local parser and by comparing
round-trip statevector probabilities. This is **not OpenQASM 3.1 validation**,
formal semantics verification, or T9 acceptance. The receipt leaves T9
`not_assessed`; any separate 3.1 gate remains pending. Global phase is irrelevant
to this study's measured probabilities and is not an export equivalence claim.

Statevector evaluation and Aer share Qiskit, so they are not independent
implementations. The analytic reference provides the separate mathematical
check. The study is small and idealized, with a hardcoded known target; it says
nothing about hardware noise, provider readiness, unknown search problems,
Erdős statements, or asymptotic computational advantage. The implementation
contains no provider imports, credential access, network calls, or installers.
