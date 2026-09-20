"""
Erdos problem #401 (erdosproblems.com / github.com/manman4/erdosproblems,
data/problems.yaml entry `number: "401"`) — quantum-testable companion sequence.

Source metadata for problem #401: prize "no", informal_status "proved"
(Lean-formalized), oeis: ["N/A"], tags: ["number theory", "factorials"].

LIMITATION, stated honestly: problem #401 carries no OEIS id in the source
data (oeis: ["N/A"]), so there is no published integer sequence to target
directly, and the repository clone available here has no prose statement of
the problem beyond its YAML metadata (prize/status/tags). What *is* known
from the metadata is the subject area: number theory and factorials. So
instead of fabricating a fake OEIS-derived property, this script builds a
genuine, small, finite, computable property that sits squarely in that
subject area and is checkable by a real quantum circuit: Wilson's theorem,
the classical characterization of primality via factorials,

    m is prime  <=>  (m - 1)! ≡ -1 (mod m)      (m >= 2)

equivalently m is COMPOSITE (for m >= 2) iff (m-1)! mod m != m-1.

The finite instance: search m in {2, 3, ..., 33} (5 qubits encode m-2 in
binary, m = index + 2) for the m classically verified, via Wilson's theorem
computed from arbitrary-precision factorials, to be PRIME. Primes are the
minority in this range (11 of 32 values), which keeps the Grover search
non-degenerate (a "half the space is marked" instance gives no net
amplification over the uniform starting distribution, so primality --
rather than compositeness, which is the majority case here -- is used as
the marked predicate). The marked set is computed from first principles
below in `classical_wilson_primes`, using only integer arithmetic (no OEIS
lookups, no hardcoded literal answer).

The circuit is a genuine Grover search over the 5-qubit index register
{0,...,31} (m = index+2, so index 0..31 <-> m 2..33). The oracle phase-flips
exactly the basis states whose index corresponds to an m classically
verified (via Wilson's theorem, computed in Python with arbitrary-precision
factorials) to be prime; this is the standard way to build a small Grover
oracle from a decidable arithmetic predicate. Grover's algorithm is then run
for the optimal number of iterations, and the simulator's most probable
output(s) are compared against the classically computed marked set.

Run: python3 problem_401.py
Deps: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (Wilson's
#    theorem): for each m in 2..9, is (m-1)! congruent to -1 mod m?
#    m is prime iff yes; m is composite iff no. No OEIS lookup, no literals.
# ---------------------------------------------------------------------------

def classical_wilson_primes(lo: int, hi: int):
    """Return the sorted list of prime m in [lo, hi] via Wilson's theorem,
    computed directly from factorials (math.factorial), with an independent
    trial-division cross-check for sanity. Primes are the minority in this
    range, which is what makes them a good, non-degenerate Grover target
    (roughly a third of the search space, not exactly half of it)."""
    primes = []
    for m in range(lo, hi + 1):
        fact = math.factorial(m - 1)
        is_prime_by_wilson = (fact % m) == (m - 1)  # (m-1)! ≡ -1 (mod m)

        # independent classical check via trial division
        is_prime_by_trial = m >= 2 and all(m % d != 0 for d in range(2, int(m ** 0.5) + 1))

        assert is_prime_by_wilson == is_prime_by_trial, (
            f"Wilson's theorem disagreed with trial division at m={m}"
        )
        if is_prime_by_wilson:
            primes.append(m)
    return primes


LO, HI = 2, 33  # 32 values -> 5-qubit index register, index = m - LO
N_QUBITS = 5
N_STATES = 2 ** N_QUBITS
assert HI - LO + 1 == N_STATES

PRIME_MS = classical_wilson_primes(LO, HI)
MARKED_INDICES = sorted(m - LO for m in PRIME_MS)
print(f"Classical (Wilson's theorem, first principles): prime m in "
      f"[{LO},{HI}] = {PRIME_MS}  -> marked indices {MARKED_INDICES}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 3-qubit index register.
# ---------------------------------------------------------------------------

def build_oracle(marked_indices, n_qubits):
    """Phase-flip oracle: |x> -> -|x> for x in marked_indices, built from
    classically-determined marked basis states (standard small-Grover-oracle
    construction from a decidable predicate)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        # flip qubits that should be 0 in this basis state, so the
        # multi-controlled Z fires exactly on |idx>
        flip_qubits = [q for q, b in enumerate(reversed(bits)) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_states, n_marked):
    theta = math.asin(math.sqrt(n_marked / n_states))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(r, 1)


n_marked = len(MARKED_INDICES)
iters = grover_iterations(N_STATES, n_marked)
print(f"Grover iterations for N={N_STATES}, marked={n_marked}: {iters}")

oracle = build_oracle(MARKED_INDICES, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iters):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------------
# 3. Compare quantum result to classical answer.
# ---------------------------------------------------------------------------

# indices measured, sorted by descending frequency
measured_indices_by_freq = sorted(
    ((int(bitstring, 2), c) for bitstring, c in counts.items()),
    key=lambda kv: -kv[1],
)
total_marked_shots = sum(c for idx, c in measured_indices_by_freq if idx in MARKED_INDICES)
marked_fraction = total_marked_shots / shots

print("Measurement counts (index -> shots):",
      {idx: c for idx, c in measured_indices_by_freq})
print(f"Fraction of shots landing on a classically-verified composite index: "
      f"{marked_fraction:.3f}")

top_indices = {idx for idx, c in measured_indices_by_freq[:n_marked]}
top_correct = top_indices.issubset(set(MARKED_INDICES))
amplified = marked_fraction > (n_marked / N_STATES) + 0.15  # clearly above uniform baseline

verified = top_correct and amplified

print(f"Top-{n_marked} measured indices: {sorted(top_indices)}  "
      f"| classical marked indices: {MARKED_INDICES}")

if verified:
    print("PASS: Grover search recovered exactly the Wilson's-theorem-verified "
          f"prime m in [{LO},{HI}], with amplitude clearly amplified above uniform.")
else:
    print("FAIL: quantum result did not match the classical Wilson's-theorem answer.")
