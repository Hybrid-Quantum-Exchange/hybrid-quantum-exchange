"""
Erdos problem #187 (erdosproblems.com) — quantum-testable instance.

Problem #187's entry in data/problems.yaml has tags
["additive combinatorics", "ramsey theory", "arithmetic progressions"] and
oeis: ["N/A"] — no OEIS sequence id is attached to this problem, so there is
no OEIS term to reproduce here. In its place we test a small, finite,
genuinely computable property from the same area (van der Waerden-type
3-term-arithmetic-progression colorings), derived and checked classically in
this script, and then found with a real Grover search circuit. This is an
honest substitute chosen because problem #187 itself has no numeric sequence
to query; the limitation is noted rather than glossed over.

Classical property tested
--------------------------
N = 8. Consider all 2-colorings of {1, ..., 8} (256 of them, one bit per
integer). A coloring is "AP-free" if no 3-term arithmetic progression
(a, a+d, a+2d) with 1 <= a, a+2d <= 8 is monochromatic under it. This is
exactly the combinatorial object behind the van der Waerden number W(2,3) = 9
(the classical theorem that at N = 9 no AP-free 2-coloring exists, while at
N = 8 AP-free colorings do exist) — the same "arithmetic progressions" /
"ramsey theory" territory as problem #187's tags.

The script first computes, purely classically/combinatorially (brute force
over all 2^8 = 256 colorings, checking all 12 three-term APs in {1..8}),
the exact set of AP-free colorings. This is the ground truth the quantum
result is checked against.

Quantum approach
-----------------
Grover search over the 8-bit coloring space (2^8 = 256 states) is built,
with the oracle marking exactly the classically-precomputed AP-free
bitstrings (implemented as multi-controlled-Z gates, one per marked
bitstring, sandwiched by X gates on the 0-bits — a standard way to encode a
classically known marked set into a Grover oracle). The optimal number of
Grover iterations is computed from the true marked count M=6 out of N=256:
floor(pi/4 * sqrt(256/6)). The circuit is run on the ideal AerSimulator and
the most frequent measured bitstring is checked against the classical
AP-free predicate directly (an independent post-check, not just membership
in the oracle's own list) and against the precomputed classical set.

PASS means: (a) the state Grover returns most often is independently
verified, by brute-force classical re-check, to be a genuine AP-free
3-coloring-avoiding-instance for N=8, and (b) it belongs to the classically
enumerated marked set.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N = 8  # integers 1..N


def three_term_aps(n):
    aps = []
    for d in range(1, n):
        a = 1
        while a + 2 * d <= n:
            aps.append((a, a + d, a + 2 * d))
            a += 1
    return aps


APS = three_term_aps(N)


def is_ap_free(bits):
    """bits[i] is the color (0/1) of integer i+1. True if no AP is monochromatic."""
    for a, b, c in APS:
        if bits[a - 1] == bits[b - 1] == bits[c - 1]:
            return False
    return True


def classical_marked_set(n):
    marked = []
    for x in range(2 ** n):
        bits = [(x >> i) & 1 for i in range(n)]
        if is_ap_free(bits):
            marked.append(x)
    return marked


MARKED = classical_marked_set(N)
M = len(MARKED)
SEARCH_SPACE = 2 ** N

print(f"Classical brute force: N={N}, {len(APS)} three-term APs checked, "
      f"{M} AP-free colorings out of {SEARCH_SPACE} total.")
assert M > 0, "expected at least one AP-free coloring for N=8 (W(2,3)=9)"


def bits_of(x, n):
    return [(x >> i) & 1 for i in range(n)]


def add_oracle_mark(qc, qubits, x, n):
    """Flip the phase of basis state |x> (n-bit) using a multi-controlled Z."""
    bits = bits_of(x, n)
    flip_qubits = [qubits[i] for i in range(n) if bits[i] == 0]
    for q in flip_qubits:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


n_qubits = N
iterations = max(1, math.floor((math.pi / 4) * math.sqrt(SEARCH_SPACE / M)))
print(f"Grover iterations used: {iterations}")

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))

diffuser = build_diffuser(n_qubits)

for _ in range(iterations):
    for x in MARKED:
        add_oracle_mark(qc, list(range(n_qubits)), x, n_qubits)
    qc.append(diffuser.to_instruction(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))
qc = qc.decompose()

sim = AerSimulator()
result = sim.run(qc, shots=2048).result()
counts = result.get_counts()

# Qiskit bit order: rightmost classical bit is qubit 0 -> reverse to match bits_of ordering.
best_bitstring = max(counts, key=counts.get)
best_x = int(best_bitstring[::-1], 2)
best_count = counts[best_bitstring]

print(f"Most frequent measured value: {best_x} (bitstring {best_bitstring}), "
      f"count {best_count}/2048")

quantum_bits = bits_of(best_x, N)
quantum_is_ap_free = is_ap_free(quantum_bits)
quantum_in_marked_set = best_x in MARKED

print(f"Independent classical re-check of measured coloring: AP-free = {quantum_is_ap_free}")
print(f"Measured value is in classically enumerated marked set: {quantum_in_marked_set}")

passed = quantum_is_ap_free and quantum_in_marked_set

if passed:
    print("PASS")
else:
    print("FAIL")
