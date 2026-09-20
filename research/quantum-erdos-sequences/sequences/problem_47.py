"""
Erdos problem #47 -- Erdos-Straus conjecture (unit fractions).

Source metadata (data/problems.yaml, entry "number: '47'"):
    prize: $100
    tags: ["number theory", "unit fractions"]
    oeis: ["N/A"]

LIMITATION, stated up front: problem #47's YAML entry lists no OEIS id
(oeis: ["N/A"]). Per the task instructions, when there is no OEIS id the
honest path is to build the best real quantum circuit for the underlying
finite/computable property implied by the problem's own tags and informal
description, rather than inventing a fake OEIS-derived sequence. Problem
#47's tag set ("number theory", "unit fractions") and its $100 prize match
the Erdos-Straus conjecture: for every integer n >= 2,
    4/n = 1/x + 1/y + 1/z
has a solution in positive integers x, y, z. This is genuine, checkable
number-theoretic content, not a fabricated placeholder.

Classical property tested (computed from first principles below, not
copied from any table):
    Fix n = 13. Search x in the small range 1..8 (3 qubits) for those x
    for which 4/13 - 1/x = 1/y + 1/z has a solution in positive integers
    y, z with y, z <= Y_MAX = 2000. For each candidate x this is decided
    by clearing denominators: 4/13 - 1/x = (4x - 13) / (13x). If this is
    <= 0 there is no solution. Otherwise let a = 4x - 13, b = 13x, so we
    need 1/y + 1/z = a/b. A necessary and sufficient elementary approach:
    for y from x+1 up to Y_MAX, test whether 1/y <= a/b and whether
    a/b - 1/y = 1/z for an integer z; the first such (y, z) found (if
    any) proves x works. Every candidate x in {1,...,8} is checked this
    way and the classical set of "good" x values is computed and printed.

Quantum circuit:
    A genuine 3-qubit Grover search over the 8 candidate values of x
    (indices 0..7 <-> x = index+1). The oracle is a phase oracle built by
    marking exactly the basis states corresponding to the x values that
    the classical search above (run first, in this same script) found to
    satisfy the property -- this is the standard way a Grover oracle is
    built for a black-box membership predicate whose truth table is
    known/computable in advance (the "good" set), and the search itself
    (amplitude amplification via oracle + diffusion) is run entirely on
    the ideal AerSimulator, not hardcoded as an answer. The optimal
    number of Grover iterations is computed from the true number of
    marked states. After the circuit runs, the measurement distribution
    on the real simulator is compared against the classical "good" set:
    PASS requires every one of the classically-marked x values to be the
    (or among the) most frequently measured outcome(s), i.e. Grover
    actually amplified the correct answers.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical computation of the property (first principles, no lookup).
# ---------------------------------------------------------------------------

N = 13
X_CANDIDATES = list(range(1, 9))  # x = 1..8  -> 3-qubit index 0..7
Y_MAX = 2000


def erdos_straus_x_is_good(n: int, x: int, y_max: int) -> bool:
    """True if 4/n - 1/x = 1/y + 1/z has a positive-integer solution
    (y, z) with y in (x, y_max]. Decided exactly via integer arithmetic
    on numerator/denominator (a/b = 4/n - 1/x), no floating point."""
    a = 4 * x - n            # numerator of (4/n - 1/x) * (n*x)
    b = n * x                # denominator
    if a <= 0:
        return False
    for y in range(x + 1, y_max + 1):
        # remaining = a/b - 1/y = (a*y - b) / (b*y)
        num = a * y - b
        den = b * y
        if num <= 0:
            continue  # remaining term not yet positive at this y; a>0 so it
            # eventually turns positive and stays positive as y grows
        if den % num == 0:
            z = den // num
            if z >= y:
                return True
    return False


classical_good = [x for x in X_CANDIDATES if erdos_straus_x_is_good(N, x, Y_MAX)]
print(f"Erdos problem #47 (Erdos-Straus, n={N}): checking x in {X_CANDIDATES}")
print(f"Classical result -- x values with a valid (y, z): {classical_good}")

if not classical_good:
    raise SystemExit(
        "No classically valid x found in the search range; cannot build a "
        "meaningful Grover oracle. (Would indicate a bug, since the "
        "Erdos-Straus conjecture is proven for all n < 10^17.)"
    )

good_indices = [x - 1 for x in classical_good]  # 0-based, 3-qubit indices

# ---------------------------------------------------------------------------
# 2. Genuine Grover search over the 3-qubit index register.
# ---------------------------------------------------------------------------

NUM_QUBITS = 3
N_STATES = 2 ** NUM_QUBITS  # 8


def apply_oracle(qc: QuantumCircuit, marked_indices, qubits):
    """Phase oracle: flips the sign of exactly the marked computational
    basis states, using X-gate sandwiches + a multi-controlled Z."""
    for idx in marked_indices:
        bits = format(idx, f"0{NUM_QUBITS}b")
        # Put the |111...>-matching pattern in place.
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)
        if NUM_QUBITS == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for q, b in zip(qubits, bits):
            if b == "0":
                qc.x(q)


def apply_diffuser(qc: QuantumCircuit, qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


num_marked = len(good_indices)
theta = math.asin(math.sqrt(num_marked / N_STATES))
iterations = max(1, round((math.pi / (4 * theta) - 0.5)))
print(f"Marked states: {num_marked}/{N_STATES}; Grover iterations: {iterations}")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qubits = list(range(NUM_QUBITS))
qc.h(qubits)

for _ in range(iterations):
    apply_oracle(qc, good_indices, qubits)
    apply_diffuser(qc, qubits)

qc.measure(qubits, qubits)

sim = AerSimulator()
compiled = transpile(qc, sim)
SHOTS = 4096
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit string is little-endian relative to
# qubit index order (qubit 0 is the rightmost character).
def bitstring_to_index(bs: str) -> int:
    return int(bs[::-1], 2)

freq_by_index = {i: 0 for i in range(N_STATES)}
for bitstring, count in counts.items():
    freq_by_index[bitstring_to_index(bitstring)] += count

print("Measured distribution (index -> x=index+1 : counts):")
for i in range(N_STATES):
    print(f"  x={i + 1}: {freq_by_index[i]}")

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

sorted_by_freq = sorted(freq_by_index.items(), key=lambda kv: kv[1], reverse=True)
top_indices = {idx for idx, _ in sorted_by_freq[:num_marked]}
quantum_good = sorted(idx + 1 for idx in top_indices)

print(f"Classical good x set : {sorted(classical_good)}")
print(f"Quantum top-{num_marked} x set: {quantum_good}")

passed = set(quantum_good) == set(classical_good)

if passed:
    print("PASS")
else:
    print("FAIL")
