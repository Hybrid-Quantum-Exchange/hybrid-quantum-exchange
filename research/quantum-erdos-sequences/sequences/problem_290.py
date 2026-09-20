"""
Erdos problem #290 (erdosproblems.com), OEIS A375081.

Erdos problem #290 concerns unit fractions / harmonic-sum denominators: for
a starting index a, look at the partial sums H(a,b) = sum_{k=a}^{b} 1/k in
lowest terms, and ask about b > a for which the denominator of H(a,b) is
*smaller* than the denominator of H(a,b-1) -- i.e. a point where adding one
more unit fraction causes cancellation that shrinks the denominator instead
of growing it. A375081 records, for each a, data about the first such
"denominator drop" point b. This script tests the a = 1 case: the ordinary
harmonic numbers H(b) = sum_{k=1}^{b} 1/k.

Classical property tested (computed here from first principles with
fractions.Fraction, not copied from OEIS):

    For b = 1..32, let D(b) = denominator of H(b) in lowest terms.
    A "drop" index is a b in [2, 32] with D(b) < D(b-1).

Computing D(1..32) directly gives the drop set {6, 18, 20, 21} (b=6 is the
classical, well-known first denominator-drop point for the harmonic
numbers -- H(5) = 137/60 has denominator 60, and H(6) = 49/20 has the
smaller denominator 20).

Quantum part: this is a finite search problem over b-1 in {0, ..., 31}
(5 qubits, N=32), so it is exactly the shape Grover's algorithm is for.
We classically compute the marked set (the drop indices, converted to
0-based b-1 values -- classical work only, no OEIS values are copied
in), build a genuine multi-controlled-Z Grover oracle over 5 qubits that
flags precisely those computational basis states, run ~2 Grover
iterations (the optimal count for 4 marked items out of 32) on the ideal
AerSimulator, and check that the measurement distribution concentrates
on the classically-known marked states.

PASS criterion: every one of the top-|marked| measured bitstrings (by
count) decodes to a b-1 value in the classically computed marked set,
and the marked-state probability mass exceeds the uniform baseline by a
wide margin.
"""

from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_drop_set(limit: int) -> list[int]:
    """Return the b in [2, limit] where denom(H(b)) < denom(H(b-1)),
    computed from first principles with exact fractions."""
    h = Fraction(0)
    denoms = []
    for b in range(1, limit + 1):
        h += Fraction(1, b)
        denoms.append(h.denominator)
    drops = []
    for i in range(1, len(denoms)):
        b = i + 1
        if denoms[i] < denoms[i - 1]:
            drops.append(b)
    return drops


N_QUBITS = 5  # 2**5 = 32 states, b - 1 for b in [1, 32]
LIMIT = 2**N_QUBITS

drops = classical_drop_set(LIMIT)
print(f"Classical harmonic-number denominator drops for b in [2,{LIMIT}]: {drops}")
assert drops == [6, 18, 20, 21], f"unexpected classical drop set: {drops}"

marked_indices = sorted(b - 1 for b in drops)  # 0-based index = b - 1
print(f"Marked (b-1) indices for Grover search: {marked_indices}")


def bits_for(index: int, n: int) -> str:
    return format(index, f"0{n}b")


def build_oracle(n: int, marked: list[int]) -> QuantumCircuit:
    """Multi-controlled-Z oracle flipping the phase of each marked basis state."""
    qc = QuantumCircuit(n, name="oracle")
    for m in marked:
        bitstr = bits_for(m, n)  # qc.mcx-style: index qubit 0 = least significant
        zero_qubits = [i for i, bit in enumerate(reversed(bitstr)) if bit == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


num_marked = len(marked_indices)
iterations = max(1, round((np.pi / 4) * np.sqrt(LIMIT / num_marked)))
print(f"N={LIMIT}, marked={num_marked}, Grover iterations={iterations}")

oracle = build_oracle(N_QUBITS, marked_indices)
diffuser = build_diffuser(N_QUBITS)

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 20000
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order: rightmost char is qubit 0 (LSB), matching bits_for().
decoded_counts = {}
for bitstr, c in counts.items():
    idx = int(bitstr, 2)
    decoded_counts[idx] = decoded_counts.get(idx, 0) + c

sorted_decoded = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
top = sorted_decoded[:num_marked]
top_indices = {idx for idx, _ in top}

marked_mass = sum(c for idx, c in decoded_counts.items() if idx in marked_indices) / shots
uniform_baseline = num_marked / LIMIT

print("Top measured (b-1) indices by count:", top)
print(f"Marked-state probability mass: {marked_mass:.4f} (uniform baseline {uniform_baseline:.4f})")

all_top_are_marked = top_indices.issubset(set(marked_indices))
enrichment_ok = marked_mass > 5 * uniform_baseline

verified = all_top_are_marked and enrichment_ok

if verified:
    print("PASS: Grover search recovered the classical harmonic-number "
          "denominator-drop set with strong amplitude amplification.")
else:
    print("FAIL: quantum result did not match the classical drop set.")

print(f"ran_ok=True verified_against_classical={verified}")
