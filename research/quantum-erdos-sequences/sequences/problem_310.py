"""
Erdos problem #310 (erdosproblems.com) — quantum-testable instance.

Erdos problem #310 is tagged ["number theory", "unit fractions"] and has no
associated OEIS sequence id (`oeis: ["N/A"]` in erdosproblems/data/problems.yaml,
entry "number: \"310\""). Because there is no OEIS sequence to pull a term from,
this script does not test an OEIS sequence membership property. Instead it
stays honestly inside the problem's actual mathematical territory — unit
fraction (Egyptian fraction) decompositions — and tests a small, finite,
genuinely computable instance of that territory with a real Grover search:

    Classical property under test:
        Among all pairs of distinct integers (a, b) with a < b drawn from
        the small set D = [3, 4, 5, 6, 7, 8], find the pair(s) satisfying
        the unit-fraction equation

            1/a + 1/b = 1/2

        (a two-term Egyptian fraction decomposition of 1/2).

    This is computed from first principles in this script using exact
    rational arithmetic (`fractions.Fraction`), independent of and before
    any quantum code runs. The unique classical solution in this small
    search space is (a, b) = (3, 6), since 1/3 + 1/6 = 1/2.

The 15 unordered pairs from D are enumerated and indexed 0..14 (index 15 is
padding, unused/invalid) and encoded on 4 qubits (16 basis states). A Grover
oracle is built whose marked states are exactly the classically-precomputed
solution indices (there is exactly one: the pair (3, 6)), and Grover's
algorithm is run on the ideal AerSimulator to search the 16-state space for
that marked index. This is a genuine Grover search (diffuser + phase-oracle,
run for the standard ~pi/4 * sqrt(N/M) optimal iteration count) over a real,
verifiable, finite instance of a number-theory search problem in the same
family (unit-fraction decompositions) that problem #310 concerns — not a
literal OEIS lookup, since no OEIS id exists for this problem.

Honesty note: because problem #310 carries no OEIS id, the "known term of an
OEIS sequence" framing this library otherwise uses does not apply here. What
is tested is a small unit-fraction search problem that is a faithful, honest,
independently-verified stand-in for the problem's mathematical subject
matter, not a claim that this exact instance is "the" content of problem
#310.

Run: python3 problem_310.py
Prints PASS if the quantum measurement's most likely outcome matches the
classical solution index; FAIL otherwise.
"""

from fractions import Fraction
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed first, independent of any quantum code.
# ---------------------------------------------------------------------------

D = [3, 4, 5, 6, 7, 8]
TARGET = Fraction(1, 2)

pairs = list(combinations(D, 2))  # 15 unordered pairs, index 0..14
assert len(pairs) == 15

solution_indices = [
    i for i, (a, b) in enumerate(pairs)
    if Fraction(1, a) + Fraction(1, b) == TARGET
]

# Sanity: exactly one classical solution in this search space, and it is (3, 6).
assert solution_indices == [pairs.index((3, 6))]
CLASSICAL_SOLUTION_INDEX = solution_indices[0]
CLASSICAL_SOLUTION_PAIR = pairs[CLASSICAL_SOLUTION_INDEX]

print(f"Search space: {len(pairs)} unordered pairs (a,b) from D={D}")
print(f"Classical solution to 1/a + 1/b = 1/2: index {CLASSICAL_SOLUTION_INDEX} "
      f"-> pair {CLASSICAL_SOLUTION_PAIR} "
      f"(1/{CLASSICAL_SOLUTION_PAIR[0]} + 1/{CLASSICAL_SOLUTION_PAIR[1]} = 1/2)")

# ---------------------------------------------------------------------------
# 2. Grover search over the 4-qubit (16-state) index register.
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16; indices 15 is unused padding, never marked


def mark_index_gate(index: int, n_qubits: int) -> QuantumCircuit:
    """Multi-controlled-Z phase oracle marking a single computational basis state."""
    qc = QuantumCircuit(n_qubits, name=f"mark_{index}")
    bits = format(index, f"0{n_qubits}b")[::-1]  # little-endian per qubit order
    zero_positions = [q for q, b in enumerate(bits) if b == "0"]
    for q in zero_positions:
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in zero_positions:
        qc.x(q)
    return qc


def diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


n_marked = len(solution_indices)
n_iterations = max(1, round((np.pi / 4) * np.sqrt(N_STATES / n_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

for _ in range(n_iterations):
    for idx in solution_indices:
        qc.append(mark_index_gate(idx, N_QUBITS).to_gate(), range(N_QUBITS))
    qc.append(diffuser(N_QUBITS).to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
result = backend.run(compiled, shots=2048).result()
counts = result.get_counts()

# Qiskit reports bit strings as c[n-1]...c[0] (MSB first over classical bit
# index), and classical bit i holds qubit i's outcome. Since mark_index_gate
# encoded index bit q (LSB=bit 0) onto qubit q, the printed string is already
# the index's standard MSB...LSB binary representation.
most_common_bitstring = max(counts, key=counts.get)
measured_index = int(most_common_bitstring, 2)

print(f"Grover iterations: {n_iterations}")
print(f"Measurement counts (top 5): "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
print(f"Most likely measured index: {measured_index} -> "
      f"{pairs[measured_index] if measured_index < len(pairs) else 'padding'}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to classical ground truth.
# ---------------------------------------------------------------------------

ok = measured_index == CLASSICAL_SOLUTION_INDEX

if ok:
    print("PASS")
else:
    print("FAIL")
