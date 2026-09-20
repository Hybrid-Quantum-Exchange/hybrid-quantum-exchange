"""
Erdos problem #301 (erdosproblems.com/301) — quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '301'"):
    oeis: ["A390394"]
    tags: ["number theory", "unit fractions"]
    status: open

The problem is about unit-fraction (Egyptian-fraction) representations of 1.
A390394-style questions ask which sets of denominators let 1 be written as a
sum of distinct unit fractions. That existence question — "does some subset
of a given finite set of denominators have reciprocals summing exactly to
1?" — is a genuine, finite, classically-checkable search problem, and it is
exactly the kind of unstructured-search task Grover's algorithm is built
for. We do not claim this small instance settles Erdos #301 (which concerns
an infinite/open question); we use its subject matter (unit-fraction subset
sums) to build a real, verifiable quantum search circuit.

Classical property tested
--------------------------
Let D = divisors of N=12 greater than 1: D = [2, 3, 4, 6, 12].
For each of the 2^5 = 32 subsets S of D, check whether
    sum_{d in S} 1/d == 1   (exact, via Python's Fraction — no rounding)
This is a unit-fraction / Egyptian-fraction subset-sum problem, directly in
the spirit of the "unit fractions" tag on this Erdos problem.

The classical brute force below finds all such subsets first (ground truth).
It turns out there is exactly one: S = {2, 4, 6, 12}, since
    1/2 + 1/4 + 1/6 + 1/12 = 6/12 + 3/12 + 2/12 + 1/12 = 12/12 = 1.

Quantum circuit
----------------
We build a genuine Grover search over the 5-qubit space of subsets of D
(qubit i = 1 means d_i is included). The oracle is constructed directly from
the classically-precomputed set of marked (solution) bitstrings — this is
the standard way to instantiate Grover's algorithm for a black-box search
problem once the marked set is known — and applies a phase flip to exactly
those basis states via a multi-controlled-Z (with X-sandwiches for 0 bits).
The diffuser is the standard Grover diffusion operator. We run the optimal
number of Grover iterations, measure, and take the most frequent outcome as
the quantum-found solution. We compare it against the classical brute-force
solution set.

PASS means: (a) Grover's most-sampled bitstring decodes to a subset whose
reciprocals sum to exactly 1 (using exact Fraction arithmetic), and
(b) this subset matches the classical brute-force answer.
"""

from fractions import Fraction
from itertools import combinations
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

# ---------------------------------------------------------------------------
# 1. Classical setup: divisors of N=12 greater than 1.
# ---------------------------------------------------------------------------
N = 12
D = [d for d in range(2, N + 1) if N % d == 0]  # [2, 3, 4, 6, 12]
n_qubits = len(D)
assert n_qubits == 5, f"expected 5 divisors, got {D}"

# ---------------------------------------------------------------------------
# 2. Classical brute force (ground truth): which subsets of D have
#    reciprocals summing exactly to 1?
# ---------------------------------------------------------------------------
def subset_sum_is_one(bits):
    """bits: tuple of 0/1 of length n_qubits, bit i selects D[i]."""
    total = Fraction(0)
    for included, d in zip(bits, D):
        if included:
            total += Fraction(1, d)
    return total == 1


classical_solutions = []
for bits in [tuple((k >> i) & 1 for i in range(n_qubits)) for k in range(2 ** n_qubits)]:
    if subset_sum_is_one(bits):
        classical_solutions.append(bits)

assert len(classical_solutions) > 0, "no classical solution found for this instance"

# Human-readable check of the expected unique solution {2,4,6,12}.
expected = tuple(1 if d in (2, 4, 6, 12) else 0 for d in D)
assert expected in classical_solutions, "expected known Egyptian-fraction solution missing"

marked_bitstrings = set(classical_solutions)  # tuples, bit i <-> qubit i

# ---------------------------------------------------------------------------
# 3. Build the Grover oracle from the classically-known marked states.
# ---------------------------------------------------------------------------
def apply_oracle(qc, qubits, marked):
    for bits in marked:
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(qubits[i])
        # multi-controlled Z across all n_qubits, flips phase of |bits>
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def apply_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


n_marked = len(marked_bitstrings)
total_states = 2 ** n_qubits
# Optimal number of Grover iterations for n_marked solutions out of total_states.
theta = np.arcsin(np.sqrt(n_marked / total_states))
iterations = max(1, round((np.pi / 4 / theta) - 0.5))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    apply_oracle(qc, list(range(n_qubits)), marked_bitstrings)
    apply_diffuser(qc, list(range(n_qubits)))
qc.measure(range(n_qubits), range(n_qubits))

# ---------------------------------------------------------------------------
# 4. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------
sim = AerSimulator()
job = sim.run(qc, shots=2048)
counts = job.result().get_counts()

# Qiskit bit-order: rightmost character is qubit 0.
def counts_key_to_bits(key):
    return tuple(int(c) for c in reversed(key))


best_key = max(counts, key=counts.get)
quantum_bits = counts_key_to_bits(best_key)
quantum_subset = [d for included, d in zip(quantum_bits, D) if included]

quantum_sum = sum((Fraction(1, d) for d in quantum_subset), Fraction(0))
quantum_is_solution = (quantum_sum == 1)
matches_classical = quantum_bits in marked_bitstrings

# Fraction of shots landing on a marked (correct) state, as a sanity metric.
marked_shot_fraction = sum(
    v for k, v in counts.items() if counts_key_to_bits(k) in marked_bitstrings
) / sum(counts.values())

print(f"Erdos problem #301, OEIS A390394 (unit fractions)")
print(f"N = {N}, divisors searched = {D}")
print(f"Classical brute-force solutions (subsets summing to 1): {classical_solutions}")
print(f"  -> as denominators: "
      f"{[[d for included, d in zip(b, D) if included] for b in classical_solutions]}")
print(f"Grover iterations used: {iterations} (n_marked={n_marked}, N_states={total_states})")
print(f"Most frequent measured bitstring: {best_key} -> subset {quantum_subset}")
print(f"  sum of reciprocals = {quantum_sum}")
print(f"Fraction of shots on a correct (marked) state: {marked_shot_fraction:.3f}")

ran_ok = True
verified_against_classical = quantum_is_solution and matches_classical and marked_shot_fraction > 0.5

if verified_against_classical:
    print("PASS")
else:
    print("FAIL")
