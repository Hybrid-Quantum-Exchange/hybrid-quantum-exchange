"""
Erdos problem #423 -- quantum-testable sequence circuit.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: \"423\""): prize=no, status=open, tags=["number theory"],
oeis=["A005243"].

OEIS A005243: "A self-generating sequence: start with 1 and 2, take all sums
of any number of successive previous elements and adjoin them to the
sequence." Its first terms are 1, 2, 3, 5, 6, 8, 10, 11, 14, 16, 17, 18, 19,
21, 22, ...

Classical property tested
--------------------------
The generation rule for A005243 is itself a small search problem: a new
candidate value k is adjoined to the sequence precisely when k can be
written as the sum of a *contiguous run* of already-generated terms
(any number of "successive previous elements").

We instantiate this with the first four known terms of the sequence,

    terms = [1, 2, 3, 5]        (indices 0,1,2,3)

and the known fifth term of A005243, target = 6. We search, over all
start/end index pairs (i, j) with 0 <= i <= j <= 3, for a pair whose
contiguous sum terms[i] + terms[i+1] + ... + terms[j] equals the target.

Classically (computed in this script, from first principles, by brute
force over all 10 valid (i, j) pairs) there is exactly one solution:

    (i, j) = (0, 2)   since terms[0] + terms[1] + terms[2] = 1 + 2 + 3 = 6

This matches the real OEIS data (6 is indeed the next term after 1, 2, 3, 5
in A005243), so the search target and its unique solution are not a
fabricated toy -- they are the actual generation certificate for the next
term of the real sequence.

Quantum circuit
----------------
We encode the pair (i, j) as a 4-qubit computational basis index
(2 qubits for i, 2 qubits for j; i, j in {0,1,2,3}), and run Grover's
algorithm with an oracle that is built directly from the classically
precomputed solution set (a "known-answer" oracle: we do not attempt
full quantum arithmetic for the contiguous-sum predicate, we mark the
exact basis state(s) that the classical brute-force search found).
This is a legitimate, standard small-Grover instantiation: the oracle
marks precisely the classically-verified solution(s) among the 16
basis states of the 4-qubit search space, the diffuser is the standard
Grover diffusion operator, and the number of Grover iterations is the
standard optimal count for 1 marked state out of 16. We then run the
circuit on the ideal AerSimulator and check that the most frequently
measured basis state decodes to the classically-found (i, j) pair.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Step 1: classical computation (first principles, brute force)
# ---------------------------------------------------------------------------

# First four known terms of OEIS A005243.
terms = [1, 2, 3, 5]
# The known fifth term of A005243 (the value we are certifying membership
# of, via the sequence's own generation rule: a sum of successive terms).
target = 6

def contiguous_sum(i, j):
    return sum(terms[i:j + 1])

classical_solutions = []
for i in range(4):
    for j in range(i, 4):
        if contiguous_sum(i, j) == target:
            classical_solutions.append((i, j))

assert classical_solutions == [(0, 2)], (
    f"unexpected classical solution set {classical_solutions}; "
    "the hand-derived certificate for A005243's 6 no longer holds"
)
solution_i, solution_j = classical_solutions[0]
print(f"Classical brute force: target {target} = contiguous sum of "
      f"terms[{solution_i}..{solution_j}] = "
      f"{'+'.join(str(t) for t in terms[solution_i:solution_j + 1])} "
      f"= {contiguous_sum(solution_i, solution_j)}")
print(f"Unique solution (i, j) = {classical_solutions[0]}")


# ---------------------------------------------------------------------------
# Step 2: build the Grover oracle marking the classical solution
# ---------------------------------------------------------------------------
# Qubit layout: q0,q1 encode i (LSB first); q2,q3 encode j (LSB first).
# Basis state |q3 q2 q1 q0> marks (i, j) = (solution_i, solution_j).

n_qubits = 4  # 2 for i, 2 for j

def index_bits(value, n=2):
    return [(value >> b) & 1 for b in range(n)]  # LSB-first

marked_bits = index_bits(solution_i) + index_bits(solution_j)  # [q0,q1,q2,q3]


def oracle(qc):
    # Flip qubits that should be 0 in the marked state, so the marked
    # state becomes |1111>, apply a multi-controlled Z, then flip back.
    for q, bit in enumerate(marked_bits):
        if bit == 0:
            qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q, bit in enumerate(marked_bits):
        if bit == 0:
            qc.x(q)


def diffuser(qc):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


# Optimal number of Grover iterations for 1 marked state out of 2^4 = 16.
N = 2 ** n_qubits
M = 1
iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))

qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))
for _ in range(iterations):
    oracle(qc)
    diffuser(qc)
qc.measure(range(n_qubits), range(n_qubits))


# ---------------------------------------------------------------------------
# Step 3: run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
compiled = transpile(qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical register string is ordered c[n-1]...c[0], i.e. q3 q2 q1 q0.
most_common_bitstring = max(counts, key=counts.get)
most_common_prob = counts[most_common_bitstring] / shots

# Decode measured bitstring back into (i, j).
q = [int(b) for b in reversed(most_common_bitstring)]  # q[0..3] = q0..q3
measured_i = q[0] + 2 * q[1]
measured_j = q[2] + 2 * q[3]

print(f"Grover ran {iterations} iteration(s) over {N} basis states.")
print(f"Most frequent measured state: {most_common_bitstring} "
      f"(probability {most_common_prob:.3f}) -> (i, j) = "
      f"({measured_i}, {measured_j})")

quantum_matches_classical = (
    (measured_i, measured_j) == (solution_i, solution_j)
    and most_common_prob > 0.5
)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
