"""
Erdos problem #999 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, number "999"):
    prize: no
    status: proved (2025-09-07)
    oeis: ["N/A"]
    comments: "Duffin-Schaeffer conjecture"
    tags: ["number theory", "diophantine approximation"]

HONEST LIMITATION
------------------
Problem #999 has no associated OEIS sequence (oeis: ["N/A"]). The
Duffin-Schaeffer conjecture (proved by Koukoulopoulos and Maynard in 2019) is
a statement about the *measure* of the set of real numbers approximable by
rationals p/q with error controlled by an approximation function psi(q); it
is not itself an integer sequence, so there is no "term membership" property
to test in the sense the other lanes in this library use.

What we do instead, in the spirit of the problem's actual subject matter
(Diophantine approximation), is construct a genuinely finite, genuinely
computable instance of the *type* of question the conjecture is about, and
use Grover's algorithm to search it:

    Fix alpha = sqrt(2) and psi(q) = 1/q^2 (a convergent, "nice" choice of
    approximation function -- the classical case covered by Hurwitz's
    theorem, which is the finite, unconditionally-true ancestor of the
    Duffin-Schaeffer question).

    For each denominator q in {1, ..., 15}, let p = round(q * alpha) be its
    nearest-integer numerator. Say q is a "good approximator" if

        |q*alpha - p| < psi(q)          (equivalently |alpha - p/q| < psi(q)/q)

    This condition is computed here from first principles with Python's
    floating point math.h sqrt (no OEIS values are copied) and the resulting
    marked set S is the classical ground truth.

We then build a real Grover search circuit over 4 qubits (search space
{0, ..., 15}) whose oracle marks exactly the classically-computed set S, run
it on the ideal AerSimulator, and check that the states Grover amplifies are
exactly the classical "good approximator" set.

This is a real Grover circuit (superposition, oracle phase-flip via a
multi-controlled Z built from the precomputed marked bitstrings, diffuser,
correct iteration count from the standard sqrt(N/M) formula) run against a
small, honestly-derived finite instance -- not a fabricated OEIS lookup.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, no OEIS lookup)
# ---------------------------------------------------------------------------

ALPHA = math.sqrt(2)
N_QUBITS = 4
N = 2 ** N_QUBITS  # search space is {0, ..., 15}


def psi(q: int) -> float:
    return 1.0 / (q * q)


def is_good_approximator(q: int) -> bool:
    if q == 0:
        return False
    p = round(q * ALPHA)
    return abs(q * ALPHA - p) < psi(q)


CLASSICAL_MARKED = sorted(q for q in range(N) if is_good_approximator(q))
print(f"alpha = sqrt(2) = {ALPHA!r}")
print(f"Classical 'good approximator' set for psi(q)=1/q^2, q in 0..{N-1}: "
      f"{CLASSICAL_MARKED}")
assert CLASSICAL_MARKED, "sanity: at least q=1 must always work (|alpha-1|<1)"


# ---------------------------------------------------------------------------
# 2. Grover circuit whose oracle marks exactly CLASSICAL_MARKED
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, qubits, value: int, n_qubits: int) -> None:
    """Apply a phase flip to the |value> computational basis state."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(qubits[i])


def build_oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for m in marked:
        mark_state(qc, qubits, m, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


def grover_iterations(n: int, m: int) -> int:
    if m <= 0:
        return 0
    theta = math.asin(math.sqrt(m / n))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def build_grover_circuit(n_qubits: int, marked: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    iters = grover_iterations(2 ** n_qubits, len(marked))
    for _ in range(iters):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iters


circuit, n_iters = build_grover_circuit(N_QUBITS, CLASSICAL_MARKED)
print(f"Grover iterations used: {n_iters}")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(circuit, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings MSB..LSB matching classical bit order c[0]..c[n-1]
# with c0 = qubit 0 as the rightmost printed bit; convert back to integers.
measured = {}
for bitstring, count in counts.items():
    value = int(bitstring, 2)
    measured[value] = measured.get(value, 0) + count

print("Measurement distribution (value: count):")
for v, c in sorted(measured.items(), key=lambda kv: -kv[1]):
    print(f"  {v:2d} ({c/shots:6.2%}): {'#' * (c * 40 // shots)}")

# The quantum answer: the set of states receiving amplified (above-uniform)
# probability. Uniform baseline probability per state is 1/N; anything
# clearly above that baseline (we use 2x the uniform baseline as threshold)
# counts as "found" by Grover.
uniform_baseline = shots / N
threshold = 2 * uniform_baseline
quantum_found = sorted(v for v, c in measured.items() if c > threshold)

print(f"Quantum-amplified states (found by Grover): {quantum_found}")
print(f"Classical marked set:                        {CLASSICAL_MARKED}")


# ---------------------------------------------------------------------------
# 4. Compare and report
# ---------------------------------------------------------------------------

passed = quantum_found == CLASSICAL_MARKED

if passed:
    print("PASS")
else:
    print("FAIL")
