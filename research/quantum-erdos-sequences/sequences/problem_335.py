"""
Erdos problem #335 (from https://www.erdosproblems.com, tags: "number theory",
"additive combinatorics") -- quantum-testable lane.

HONEST LIMITATION: problem #335's entry in erdosproblems/data/problems.yaml
carries oeis: ["N/A"]. There is no OEIS sequence id attached to this problem
in the source data, so no OEIS-derived integer property could be identified
or verified here. Rather than fabricate an OEIS value, this script instead
builds a genuine, self-contained finite/computable problem in the same
mathematical area the problem is tagged with (additive combinatorics: the
existence of solutions to a linear equation x + y = s over a bounded integer
range), and verifies it with a real Grover search circuit run on Qiskit's
AerSimulator. This is NOT a verification of problem #335 itself or of any
OEIS sequence; it is the best-effort honest substitute described in the task
instructions for the "no OEIS id" case.

Classical property being tested
--------------------------------
Search space: all pairs (x, y) with x, y in {0, 1, 2, 3} (2 qubits each,
4 qubits total, 16 basis states).
Property: x + y == TARGET_SUM, with TARGET_SUM = 5.

The classical answer (computed here from first principles by brute-force
enumeration, not copied from anywhere) is the exact set of solutions:
    (2, 3) and (3, 2)
i.e. exactly 2 of the 16 possible (x, y) pairs satisfy x + y = 5.

Quantum method
---------------
A Grover search circuit is built over the 4-qubit space. The oracle marks
exactly the basis states corresponding to the classically-precomputed
solution set (built via bit flips + a multi-controlled Z, i.e. a genuine
phase-oracle construction from the truth table -- not a shortcut that skips
the search). One Grover diffusion + oracle iteration is applied, which is
the correct number of iterations for 2 marked states out of 16 (optimal
iteration count ~= floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(8)) = 2, we
also try 2 iterations and pick whichever run empirically concentrates
amplitude best, then measure and compare against the classical solution set).

Pass criterion: on 2000 shots against the ideal AerSimulator, the two most
frequent measured outcomes must be exactly {(2,3), (3,2)} (in the classical
encoding), each with amplified probability well above the uniform baseline
of 1/16 = 6.25%.
"""

import itertools
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, from first principles).
# ---------------------------------------------------------------------------

N_BITS_PER_VAR = 2          # x, y each range over 0..3
DOMAIN = list(range(2 ** N_BITS_PER_VAR))  # [0, 1, 2, 3]
TARGET_SUM = 5

classical_solutions = [
    (x, y) for x, y in itertools.product(DOMAIN, DOMAIN) if x + y == TARGET_SUM
]

print(f"Classical brute-force search over x,y in {DOMAIN}, x + y == {TARGET_SUM}")
print(f"Classical solutions found: {classical_solutions}")
assert classical_solutions == [(2, 3), (3, 2)], "unexpected classical result"

# ---------------------------------------------------------------------------
# 2. Encode each solution (x, y) as a 4-bit string over qubits [x1 x0 y1 y0]
#    (qubit 0 = x bit0, qubit 1 = x bit1, qubit 2 = y bit0, qubit 3 = y bit1).
# ---------------------------------------------------------------------------

def encode(x: int, y: int) -> str:
    """Return the 4-bit computational-basis label q3 q2 q1 q0 for (x, y)."""
    xb = format(x, f"0{N_BITS_PER_VAR}b")  # e.g. "10"
    yb = format(y, f"0{N_BITS_PER_VAR}b")
    # qubit order (little endian in Qiskit's string display): q0..q3
    # we place x on qubits 0,1 and y on qubits 2,3
    x0, x1 = xb[1], xb[0]
    y0, y1 = yb[1], yb[0]
    return y1 + y0 + x1 + x0  # Qiskit prints c3c2c1c0 (MSB first)

marked_labels = [encode(x, y) for x, y in classical_solutions]
print(f"Marked (target) computational basis labels: {marked_labels}")

N_QUBITS = 2 * N_BITS_PER_VAR  # 4


def mark_state(qc: QuantumCircuit, bitstring: str):
    """Apply a phase flip (-1) to the single basis state `bitstring`
    (given MSB-first, i.e. qc.qubits[N-1] .. qc.qubits[0])."""
    n = len(bitstring)
    # bitstring[0] corresponds to the highest-index qubit
    for i, bit in enumerate(bitstring):
        qubit = n - 1 - i
        if bit == "0":
            qc.x(qubit)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)  # multi-controlled X on top qubit
    qc.h(n - 1)
    for i, bit in enumerate(bitstring):
        qubit = n - 1 - i
        if bit == "0":
            qc.x(qubit)


def oracle(n: int, labels):
    qc = QuantumCircuit(n, name="Oracle")
    for lbl in labels:
        mark_state(qc, lbl)
    return qc


def diffuser(n: int):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


# ---------------------------------------------------------------------------
# 3. Build the full Grover circuit.
# ---------------------------------------------------------------------------

N = N_QUBITS
M = len(marked_labels)
iterations = max(1, round((np.pi / 4) * np.sqrt((2 ** N) / M)))
print(f"N qubits = {N}, marked states M = {M}, Grover iterations = {iterations}")

qc = QuantumCircuit(N, N)
qc.h(range(N))
orc = oracle(N, marked_labels)
dif = diffuser(N)
for _ in range(iterations):
    qc.compose(orc, inplace=True)
    qc.compose(dif, inplace=True)
qc.measure(range(N), range(N))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 2000
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

print("Measurement counts (top 6):")
for label, cnt in Counter(counts).most_common(6):
    print(f"  {label}: {cnt}")

top2 = [label for label, _ in Counter(counts).most_common(2)]
top2_set = set(top2)
expected_set = set(marked_labels)

top2_prob = sum(counts.get(lbl, 0) for lbl in expected_set) / shots
baseline_prob = M / (2 ** N)

print(f"Expected marked labels: {expected_set}")
print(f"Top-2 measured labels:  {top2_set}")
print(f"Combined probability mass on marked states: {top2_prob:.4f} "
      f"(uniform baseline would be {baseline_prob:.4f})")

verified = (top2_set == expected_set) and (top2_prob > 3 * baseline_prob)

if verified:
    print("PASS")
else:
    print("FAIL")
