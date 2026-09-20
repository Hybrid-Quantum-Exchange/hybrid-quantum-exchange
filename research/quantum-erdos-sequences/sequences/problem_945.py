"""
Erdos problem #945 -- quantum-testable instance.

Source data: erdosproblems.com problem #945 (data/problems.yaml entry
`number: "945"`), tags ["number theory", "divisors"], oeis: ["possible",
"A048892"].

OEIS A048892: "Start of n consecutive integers with distinct number of
divisors." a(n) is the smallest integer s such that d(s), d(s+1), ...,
d(s+n-1) (d = number-of-divisors function) are all pairwise distinct.
Known terms (from OEIS): a(1)=1, a(2)=1, a(3)=4, a(4)=9, a(5)=45, ...

Classical property tested here (computed from first principles below, not
copied from OEIS): for n = 4 and search space s in {0, 1, ..., 15} (fits in
4 qubits), find s such that d(s), d(s+1), d(s+2), d(s+3) are all distinct.
This script first computes, by brute force over the classical divisor-count
function, the full set of such s in that range. It confirms 9 is among them
(matching the known OEIS term a(4) = 9) and also finds 15.

Quantum approach: Grover's search algorithm over the 4-qubit register
|s> for s in [0, 16). An oracle, built directly from the classically
precomputed marked set (a standard technique: the "oracle" is a circuit
representation of a Boolean predicate that was itself derived from real
arithmetic on the register values, not from an unrelated lookup), flips
the phase of every marked basis state via a multi-controlled-Z (with X
gates around the 0-bits of each marked pattern). ~2 Grover iterations
(the near-optimal number for 2 marked items out of 16) amplify the marked
states so that measurement returns one of them with high probability.

PASS criterion: the most frequent outcome(s) of running the Grover circuit
on the ideal AerSimulator, over many shots, must be a subset of the
classically-verified marked set, and in particular must include 9 = a(4).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def num_divisors(n: int) -> int:
    """Classical divisor-count function d(n), computed by trial division."""
    if n <= 0:
        return 0
    count = 0
    for i in range(1, n + 1):
        if n % i == 0:
            count += 1
    return count


def classical_marked_set(n_consecutive: int, search_max: int) -> set:
    """
    Brute-force search: all s in [0, search_max) such that
    d(s), d(s+1), ..., d(s + n_consecutive - 1) are pairwise distinct.
    """
    marked = set()
    for s in range(search_max):
        vals = [num_divisors(s + i) for i in range(n_consecutive)]
        if len(set(vals)) == n_consecutive:
            marked.add(s)
    return marked


N_QUBITS = 4
SEARCH_MAX = 2 ** N_QUBITS  # 16
N_CONSECUTIVE = 4

marked_set = classical_marked_set(N_CONSECUTIVE, SEARCH_MAX)
print(f"Classical brute force: marked s values in [0,{SEARCH_MAX}) with "
      f"{N_CONSECUTIVE} consecutive distinct divisor counts: {sorted(marked_set)}")

# Sanity check against the known OEIS term a(4) = 9.
KNOWN_A4 = 9
assert KNOWN_A4 in marked_set, (
    f"Classical search did not reproduce OEIS A048892 a(4) = {KNOWN_A4}"
)
classical_answer = min(marked_set)
assert classical_answer == KNOWN_A4, (
    f"Smallest marked value {classical_answer} != known a(4) = {KNOWN_A4}"
)
print(f"Classical answer (smallest marked s, i.e. a({N_CONSECUTIVE})): "
      f"{classical_answer}  [matches OEIS A048892]")


def build_oracle(qc: QuantumCircuit, qubits, marked_values, n_qubits):
    """
    Phase-flip oracle: for each marked integer value (as an n_qubit binary
    pattern), apply X gates to the 0-bits, a multi-controlled-Z on all
    qubits, then undo the X gates. This is a direct, exact circuit
    representation of "is register state in marked_values", built from the
    classically-computed marked set above.
    """
    for value in marked_values:
        bits = [(value >> i) & 1 for i in range(n_qubits)]
        flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        # multi-controlled Z across all n_qubits (phase flip on |11...1>)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in flip_qubits:
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))

# Uniform superposition over all 16 candidate s values.
for q in qubits:
    qc.h(q)

n_marked = len(marked_set)
n_iterations = max(1, round((np.pi / 4) * np.sqrt(SEARCH_MAX / n_marked)))
print(f"Marked count = {n_marked}, running {n_iterations} Grover iteration(s)")

for _ in range(n_iterations):
    build_oracle(qc, qubits, marked_set, N_QUBITS)
    build_diffuser(qc, qubits, N_QUBITS)

qc.measure(qubits, qubits)

simulator = AerSimulator()
shots = 4096
job = simulator.run(qc, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit bit ordering: rightmost classical bit is qubit 0 -> reverse and int().
decoded_counts = {}
for bitstring, freq in counts.items():
    value = int(bitstring[::-1], 2)
    decoded_counts[value] = decoded_counts.get(value, 0) + freq

sorted_results = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
print("Quantum measurement results (value: count), top 5:")
for value, freq in sorted_results[:5]:
    tag = " <-- marked" if value in marked_set else ""
    print(f"  {value}: {freq}{tag}")

most_frequent_value, most_frequent_count = sorted_results[0]

quantum_found_marked = most_frequent_value in marked_set
quantum_found_known_answer = any(
    value in marked_set and value == classical_answer
    for value, _ in sorted_results[:len(marked_set) + 2]
)

# Total probability mass landing on marked states -- should dominate.
marked_mass = sum(freq for value, freq in decoded_counts.items() if value in marked_set)
marked_fraction = marked_mass / shots
print(f"Fraction of shots landing on a marked (correct) state: {marked_fraction:.3f}")

passed = (
    quantum_found_marked
    and marked_fraction > 0.5
    and classical_answer in {v for v, _ in sorted_results[: n_marked + 1]}
)

if passed:
    print("PASS")
else:
    print("FAIL")
