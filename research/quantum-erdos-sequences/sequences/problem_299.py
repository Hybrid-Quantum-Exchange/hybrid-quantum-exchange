"""
Erdos problem #299 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
number: "299").

Source metadata for this problem lists no OEIS sequence id (oeis: ["N/A"]) and
no title/statement text is present in the read-only clone of the problems
dataset used here -- only status ("disproved (Lean)") and tags:
["number theory", "unit fractions"]. Because there is no OEIS id to build a
genuine "sequence membership" test from, this script is the documented
fallback described in the task: a best-honest-effort quantum circuit built
directly from the problem's *tags* (unit fractions / Egyptian fractions),
not from any fabricated OEIS value.

Classical property under test
------------------------------
The classical, finite, computable property chosen from the "unit fractions"
tag is a small Egyptian-fraction search:

    Among c in {1, 2, ..., 15} (a 4-qubit register), find the unique c such
    that
        1/2 + 1/3 + 1/c == 1
    exactly (as an exact rational, via fractions.Fraction -- no floating
    point is used).

This is a real, checkable instance of a unit-fraction decomposition
question (1 as a sum of three unit fractions), the same flavor of question
that "unit fractions" Erdos problems concern. The classical answer is
derived from first principles in this script (brute-force exact rational
search over the 4-bit space), not copied from any source:

    1/2 + 1/3 + 1/6 = 3/6 + 2/6 + 1/6 = 6/6 = 1   =>  c = 6 is the unique hit.

Quantum construction
---------------------
Grover's algorithm is used to search the 4-qubit register (16 basis states,
c = 0..15, with c = 0 treated as "no match") for the unique marked value
c = 6. The oracle is built directly from the classical truth table computed
above (a legitimate way to build a Grover oracle for an arbitrary decidable
predicate over a small register: evaluate the predicate classically for
every basis state, then flip the phase of exactly the marked states with a
multi-controlled-Z, closing NOT gates on the bits that must be 0). One
Grover iteration is optimal for N=16, M=1 marked item
(iterations = round(pi/4 * sqrt(N/M)) = 3, computed below), and the ideal
AerSimulator is run to confirm the marked state c = 6 is the overwhelmingly
most likely measurement outcome.

PASS/FAIL: the script prints PASS iff the most frequent measured 4-bit
string, interpreted as an integer, equals the classically-derived unique
solution c = 6.

Limitation note (honesty per task instructions): this problem's own OEIS
id and full statement were not recoverable from the available data
(oeis: ["N/A"], no title/statement field in the dataset clone), so the
tested property is derived from the problem's tags rather than from an
OEIS sequence definition specific to problem #299 itself.
"""

import math
from fractions import Fraction

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, exact rational arithmetic).
# ---------------------------------------------------------------------------

N_QUBITS = 4
N_VALUES = 2 ** N_QUBITS  # 16 -> c ranges over 0..15


def satisfies_unit_fraction_identity(c: int) -> bool:
    """True iff 1/2 + 1/3 + 1/c == 1 exactly (c=0 is undefined -> False)."""
    if c <= 0:
        return False
    return Fraction(1, 2) + Fraction(1, 3) + Fraction(1, c) == 1


classical_matches = [c for c in range(N_VALUES) if satisfies_unit_fraction_identity(c)]
assert classical_matches == [6], (
    f"expected the unique classical solution c=6, computed {classical_matches}"
)
classical_answer = classical_matches[0]
print(f"Classical search over c in 0..{N_VALUES - 1}: matches = {classical_matches}")
print(f"Classical answer (unique c with 1/2 + 1/3 + 1/c = 1): {classical_answer}")


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle from the classical truth table.
# ---------------------------------------------------------------------------

def add_mark_state(qc: QuantumCircuit, qubits, value: int, n_qubits: int) -> None:
    """Flip the phase of |value> using a multi-controlled Z, built by
    surrounding the control bits that must be 0 with X gates."""
    bits = format(value, f"0{n_qubits}b")[::-1]  # bits[i] = value of qubits[i]
    zero_positions = [qubits[i] for i, b in enumerate(bits) if b == "0"]

    for q in zero_positions:
        qc.x(q)

    if n_qubits == 1:
        qc.z(qubits[0])
    elif n_qubits == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])

    for q in zero_positions:
        qc.x(q)


def build_oracle(n_qubits: int, marked_values) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    qubits = list(range(n_qubits))
    for v in marked_values:
        add_mark_state(qc, qubits, v, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)
    return qc


# ---------------------------------------------------------------------------
# 3. Assemble the full Grover circuit.
# ---------------------------------------------------------------------------

num_marked = len(classical_matches)
iterations = max(1, round((math.pi / 4) * math.sqrt(N_VALUES / num_marked)))
print(f"Grover iterations used: {iterations} (N={N_VALUES}, M={num_marked})")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, classical_matches)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order is little-endian in the returned bitstring (qubit 0 is the
# rightmost character), matching how add_mark_state/measure indexed qubits.
def bitstring_to_int(bitstring: str) -> int:
    return int(bitstring[::-1], 2)

int_counts = {}
for bitstring, count in counts.items():
    int_counts[bitstring_to_int(bitstring)] = int_counts.get(bitstring_to_int(bitstring), 0) + count

most_likely = max(int_counts, key=int_counts.get)
most_likely_prob = int_counts[most_likely] / shots

print(f"Measurement counts (as integers): {dict(sorted(int_counts.items()))}")
print(f"Most likely measured value: {most_likely} (probability ~{most_likely_prob:.3f})")

quantum_matches_classical = most_likely == classical_answer

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
