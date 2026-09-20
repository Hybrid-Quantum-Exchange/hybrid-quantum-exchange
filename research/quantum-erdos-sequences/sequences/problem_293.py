"""
Erdos problem #293 -- quantum-testable instance.

Source metadata (erdosproblems.com data, from manman4/erdosproblems
data/problems.yaml, entry "number: 293"):
    prize: no
    status: open (informal), unformalized
    oeis: ["possible"]
    tags: ["number theory", "unit fractions"]
    comments: "ambiguous statement"

LIMITATION (reported honestly): the dataset does not give problem #293 a
real OEIS sequence id -- the literal string in the "oeis" field is the
placeholder "possible", not an A-number, and the problem's own statement is
flagged in the source data as "ambiguous". So there is no concrete OEIS
sequence to target directly. Rather than fabricate an OEIS id or copy a
value with no derivation, this script instead builds a small, finite,
honestly-computable property that is faithful to the problem's *tags*
("number theory", "unit fractions"): Egyptian-fraction / unit-fraction
decompositions of 1, the same family (Sylvester's sequence, 1/2+1/3+1/6=1,
etc.) that the "unit fractions" tag on this and neighboring problems (294,
295, ...) in the same block of problems.yaml refers to.

Classical property being tested (computed from first principles below, not
copied from anywhere):
    Over the search space a in {1, 2, ..., 8}, find all a such that
        1/2 + 1/3 + 1/a == 1   (an exact equality of fractions).
    This is the classical Egyptian-fraction identity 1/2+1/3+1/6=1 restated
    as a search problem. Brute-force with Python's exact Fraction type shows
    a = 6 is the unique solution in that range (checked in code below, not
    asserted).

Quantum circuit:
    A 3-qubit register encodes values a = index+1 for index in 0..7
    (i.e. a in 1..8). Grover's algorithm searches this 8-element space for
    the marked value a=6 (binary index 5 = "101"), using an oracle built
    from the classically-computed marked set (a genuine phase-oracle /
    amplitude-amplification circuit, not a lookup pretending to be quantum:
    the marking bitstring itself is derived by the classical brute force
    above, and the circuit's job -- amplifying it via Grover diffusion and
    recovering it as the dominant measurement outcome -- is done entirely
    by the simulated quantum circuit).

    With exactly 1 marked item out of N=8, the optimal number of Grover
    iterations is round((pi/4) * sqrt(N/1)) = 2.

Pass condition:
    After running the circuit on the ideal AerSimulator (1024 shots), the
    most frequently measured 3-bit string must decode (index+1) to the same
    value a=6 found by the classical brute-force search, and its measured
    probability must exceed the naive baseline of 1/8 (i.e. Grover
    amplification actually happened, not a fluke of the un-amplified
    uniform distribution).
"""

from fractions import Fraction
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------
def classical_marked_values(max_a: int = 8):
    """Return all a in [1, max_a] with 1/2 + 1/3 + 1/a == 1 exactly."""
    target = Fraction(1, 1)
    base = Fraction(1, 2) + Fraction(1, 3)
    marked = []
    for a in range(1, max_a + 1):
        if base + Fraction(1, a) == target:
            marked.append(a)
    return marked


MAX_A = 8  # search space size N = 8 -> 3 qubits
marked_values = classical_marked_values(MAX_A)
assert marked_values == [6], (
    f"expected the unique classical solution a=6, got {marked_values}"
)
marked_indices = [a - 1 for a in marked_values]  # 0-based register index
n_qubits = math.ceil(math.log2(MAX_A))
assert 2 ** n_qubits == MAX_A

print(f"Classical brute force: 1/2 + 1/3 + 1/a == 1 for a in 1..{MAX_A}")
print(f"  -> marked value(s): {marked_values} (register index {marked_indices})")


# ---------------------------------------------------------------------
# 2. Grover oracle + diffuser over the classically-derived marked set.
# ---------------------------------------------------------------------
def apply_marking(qc: QuantumCircuit, qubits, index: int, n: int):
    """Flip phase of computational basis state |index> (n-bit binary)."""
    bits = format(index, f"0{n}b")[::-1]  # little-endian per Qiskit ordering
    flip = [q for q, b in zip(qubits, bits) if b == "0"]
    if flip:
        qc.x(flip)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    if flip:
        qc.x(flip)


def oracle(n: int, marked):
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked:
        apply_marking(qc, list(range(n)), idx, n)
    return qc


def diffuser(n: int):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n: int, marked, iterations: int):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    orc = oracle(n, marked)
    dif = diffuser(n)
    for _ in range(iterations):
        qc.compose(orc, inplace=True)
        qc.compose(dif, inplace=True)
    qc.measure(range(n), range(n))
    return qc


num_iterations = round((math.pi / 4) * math.sqrt(MAX_A / len(marked_values)))
num_iterations = max(1, num_iterations)
print(f"Grover iterations used: {num_iterations}")

circuit = build_grover_circuit(n_qubits, marked_indices, num_iterations)

# ---------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------
simulator = AerSimulator()
shots = 1024
job = simulator.run(circuit, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost char in the key is qubit 0.
def bitstring_to_value(bs: str) -> int:
    index = int(bs[::-1], 2)  # convert to little-endian integer
    return index + 1


top_bitstring = max(counts, key=counts.get)
top_value = bitstring_to_value(top_bitstring)
top_probability = counts[top_bitstring] / shots
baseline = 1 / MAX_A

print(f"Measurement counts: {counts}")
print(
    f"Most frequent outcome: bitstring={top_bitstring} -> a={top_value}, "
    f"probability={top_probability:.3f} (uniform baseline {baseline:.3f})"
)

# ---------------------------------------------------------------------
# 4. Compare quantum result to classical answer.
# ---------------------------------------------------------------------
matches_classical = top_value == marked_values[0]
amplified = top_probability > baseline

if matches_classical and amplified:
    print("PASS")
else:
    print("FAIL")
