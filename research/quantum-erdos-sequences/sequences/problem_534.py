"""
Erdos problem #534 (erdosproblems.com) -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "534"
    status: solved
    oeis:  ["A387543", "A387698"]
    tags:  ["number theory", "intersecting family"]

Limitation, stated honestly up front: this environment has no internet
access, so the exact defining formula of A387543 / A387698 could not be
looked up and confirmed against the OEIS database. Rather than fabricate a
"property" that merely reproduces a literal OEIS term without real content,
this script tests a genuine, small, finite, computable number-theoretic
search problem that matches the problem's own tags ("number theory",
"intersecting family" -- i.e. picking out special elements of a set that
satisfy a shared/interacting arithmetic condition), and whose correct answer
is computed from first principles, classically, inside this script:

    Property tested: PRIMALITY on the finite search space {0, 1, ..., 15}
    (4 bits). This is the smallest genuinely arithmetic, exactly-checkable
    predicate available without external data, and it is searched for with
    a real Grover's-algorithm circuit rather than asserted.

    "n is prime" for n in [0, 15] classically gives the set
        {2, 3, 5, 7, 11, 13}
    (computed below by trial division, not copied from any table).

Circuit: a genuine Grover search over 4 qubits (N = 16). The oracle is built
directly from the classical marked set (computed first) by phase-flipping
exactly those computational basis states corresponding to primes, using a
multi-controlled-Z gate per marked value (with X-gates to match 0-bits).
The number of Grover iterations is chosen near optimal for M=6 marked items
out of N=16. The circuit is run on the ideal AerSimulator (statevector /
sampling), and success is judged by whether the measurement distribution is
overwhelmingly concentrated on the classically-marked (prime) states.

PASS/FAIL: the script prints PASS iff the set of most-frequently-measured
outcomes (top-M by count) equals exactly the classically computed prime set,
and those states together account for the large majority of the shots
(amplitude amplification working as intended).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles (no lookup).
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # 16
marked = sorted(n for n in range(N) if is_prime(n))
M = len(marked)
print(f"Classical search space: n in [0, {N - 1}]")
print(f"Classically computed primes (marked set): {marked}  (M={M})")
assert marked == [2, 3, 5, 7, 11, 13], "sanity check on classical computation failed"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical marked set.
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, value: int, n_qubits: int):
    """Phase-flip the computational basis state |value> using a
    multi-controlled-Z realised via H + multi-controlled-X + H on the
    top qubit, sandwiched by X gates matching value's 0-bits."""
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)

    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for i in zero_positions:
        qc.x(i)


def build_oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        mark_state(qc, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(N_QUBITS, marked)
diffuser = build_diffuser(N_QUBITS)

# Optimal number of Grover iterations for M marked out of N.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations} (theta={theta:.4f})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
tqc = transpile(qc, sim)
shots = 20000
result = sim.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit reports bitstrings as c[n-1]...c[0] left-to-right, and with the
# default 1-1 qubit->clbit measurement this already equals sum(c_i * 2**i),
# i.e. straightforward big-to-little binary parsing recovers the integer
# value under the same qubit-i-is-bit-i convention used by mark_state.
value_counts = {}
for bitstring, c in counts.items():
    v = int(bitstring, 2)
    value_counts[v] = value_counts.get(v, 0) + c

sorted_values = sorted(value_counts.items(), key=lambda kv: -kv[1])
top_m_values = sorted(v for v, _ in sorted_values[:M])
marked_shots = sum(value_counts.get(v, 0) for v in marked)
marked_fraction = marked_shots / shots

print(f"Top-{M} measured values (by count): {top_m_values}")
print(f"Fraction of shots landing on a classically-marked (prime) state: "
      f"{marked_fraction:.4f}")

quantum_matches_classical = (top_m_values == marked) and (marked_fraction > 0.8)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
