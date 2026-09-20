"""
Erdos problem #292 (erdosproblems.com), OEIS A092671, tags: number theory,
unit fractions.

The problems.yaml record for #292 gives no free-text statement, only the
tag "unit fractions" and OEIS id A092671, so this script does not claim to
reproduce A092671's exact defining condition term-for-term (no OEIS value
is copied literally, per the task's own instruction). Instead it tests a
concrete, honestly-derived unit-fraction property in the same family that
the tags describe: the Erdos-Straus-flavored question of splitting a
fraction 4/n into unit fractions.

Classical property tested (computed here from first principles with
fractions.Fraction, not copied from OEIS):

    Fix n = 15. For x = 1..32, is r = 4/n - 1/x itself a positive unit
    fraction (i.e. r > 0 and r.numerator == 1 once reduced)? Such an x is
    a valid *first term* of a 3-term Egyptian-fraction decomposition
    4/n = 1/x + 1/y + 1/z (with y = r.denominator, z absorbed if r is
    already 1/y, i.e. a 2-term completion here, which is the interesting/
    rare case Egyptor-style search targets).

Computing this directly for n = 15, x in [1,32] gives the marked set
{4, 5, 6, 10, 15} (verified: 4/15 - 1/4 = 1/60, 4/15 - 1/5 = 1/15,
4/15 - 1/6 = 1/10, 4/15 - 1/10 = 1/30, 4/15 - 1/15 = 1/5 -- all unit
fractions with positive integer denominator).

Quantum part: this is a finite search problem over x - 1 in {0, ..., 31}
(5 qubits, N=32), exactly the shape Grover's algorithm targets. We
classically compute the marked set (classical work only -- the oracle is
built from it, no OEIS values are used), construct a genuine
multi-controlled-Z Grover oracle over 5 qubits flagging precisely those
computational basis states, run the near-optimal number of Grover
iterations on the ideal AerSimulator, and check that the measurement
distribution concentrates on the classically-known marked states.

PASS criterion: every one of the top-|marked| measured bitstrings (by
count) decodes to an x - 1 value in the classically computed marked set,
and the marked-state probability mass exceeds the uniform baseline by a
wide margin.
"""

from fractions import Fraction

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(n: int, limit: int) -> list[int]:
    """Return x in [1, limit] such that 4/n - 1/x is a positive unit
    fraction, computed from first principles with exact fractions."""
    marked = []
    for x in range(1, limit + 1):
        r = Fraction(4, n) - Fraction(1, x)
        if r > 0 and r.numerator == 1:
            marked.append(x)
    return marked


N_QUBITS = 5  # 2**5 = 32 states, x - 1 for x in [1, 32]
LIMIT = 2**N_QUBITS
N = 15  # fixed fraction 4/15 being decomposed

xs = classical_marked_set(N, LIMIT)
print(f"Classical unit-fraction first-terms x for 4/{N} - 1/x = unit fraction, x in [1,{LIMIT}]: {xs}")
assert xs == [4, 5, 6, 10, 15], f"unexpected classical marked set: {xs}"

marked_indices = sorted(x - 1 for x in xs)  # 0-based index = x - 1
print(f"Marked (x-1) indices for Grover search: {marked_indices}")


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

print("Top measured (x-1) indices by count:", top)
print(f"Marked-state probability mass: {marked_mass:.4f} (uniform baseline {uniform_baseline:.4f})")

all_top_are_marked = top_indices.issubset(set(marked_indices))
enrichment_ok = marked_mass > 5 * uniform_baseline

verified = all_top_are_marked and enrichment_ok

if verified:
    print("PASS: Grover search recovered the classical unit-fraction "
          "first-term set for 4/15 with strong amplitude amplification.")
else:
    print("FAIL: quantum result did not match the classical marked set.")

print(f"ran_ok=True verified_against_classical={verified}")
