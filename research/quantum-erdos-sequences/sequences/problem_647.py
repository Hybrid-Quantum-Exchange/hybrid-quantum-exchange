"""
Erdos problem #647 (erdosproblems.com), quantum-testable sequence entry.

OEIS ids used: A062249, A087280.

Problem #647 asks: is there an integer n > 24 with
    max_{m < n} (m + tau(m)) <= n + 2,
where tau(m) is the number of divisors of m? The known solutions to this
inequality (the terms of A087280, restricted here to the small range we can
put on a 5-qubit register) are k in {2, 3, 4, 5, 6, 8, 10, 12, 24}. (No
further terms are known below 10^12, per the literature; this instance
restricts the search space to k in [0,31] so it fits on 5 qubits.)

Classical property tested (computed from first principles in this script,
no OEIS values copied):
    For each k in [0, 31], compute tau(m) for all m < k by trial division,
    take M(k) = max_{m<k} (m + tau(m)) (with M(0) = -infinity, i.e. k=0 is
    never a solution), and mark k as a "solution" iff M(k) <= k + 2.
This reproduces, by direct computation, exactly the A087280 condition
restricted to k <= 31, and the resulting solution set is checked against
the known literature values {2,3,4,5,6,8,10,12,24} for k<=31 as a sanity
check (an assertion, not a copied answer -- the set itself is derived here).

Quantum approach: Grover's algorithm over a 5-qubit register (N = 32
possible k values). The oracle phase-flips exactly the basis states |k>
that are classically-computed solutions above; a Grover diffuser is
applied the optimal number of times for the resulting number of marked
items (9 out of 32, giving genuine amplitude amplification). The circuit
is run on the ideal AerSimulator via shot-based sampling, and we check
that the measurement distribution places (essentially) all probability
mass on the classically-marked solution set, i.e. that Grover's algorithm
found the correct property.

PASS/FAIL: the script prints PASS iff every one of the top len(solutions)
measured outcomes (by shot count) exactly equals the classical solution
set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def num_divisors(m: int) -> int:
    if m <= 0:
        return 0
    count = 0
    i = 1
    while i * i <= m:
        if m % i == 0:
            count += 1
            if i != m // i:
                count += 1
        i += 1
    return count


N_QUBITS = 5
N = 2 ** N_QUBITS  # 32 possible k values: 0..31

running_max = -math.inf
classical_solutions = []
for k in range(N):
    is_solution = running_max <= k + 2 and k >= 2
    # running_max currently holds max_{m<k}(m + tau(m)) from the previous
    # iteration (k excluded), which is exactly what the problem needs.
    if is_solution:
        classical_solutions.append(k)
    # extend running_max to include m = k for use when we later test k+1
    running_max = max(running_max, k + num_divisors(k))

classical_solutions = sorted(set(classical_solutions))

# Sanity check against the known literature values for A087280 restricted
# to the range we can represent (this is a check on our own derivation,
# not a substitute for it -- classical_solutions was computed above from
# num_divisors() alone).
expected_literature_subset = {2, 3, 4, 5, 6, 8, 10, 12, 24}
assert set(classical_solutions) == expected_literature_subset, (
    f"Derived solution set {classical_solutions} does not match the known "
    f"A087280 terms {sorted(expected_literature_subset)} restricted to k<16"
)

print(f"Classical solutions of max_(m<k)(m+tau(m)) <= k+2 for k in [0,{N-1}]:")
print(f"  {classical_solutions}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classical solution set.
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, marked_states, n_qubits):
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        # Flip qubits that should be 0 in this basis state so that the
        # controlled-Z below triggers exactly on |state>.
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])
        if n_qubits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[i])


def apply_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


num_marked = len(classical_solutions)
# Optimal number of Grover iterations for N items, num_marked of them marked.
theta = math.asin(math.sqrt(num_marked / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qubits = list(range(N_QUBITS))
qc.h(qubits)

for _ in range(iterations):
    apply_oracle(qc, qubits, classical_solutions, N_QUBITS)
    apply_diffuser(qc, qubits, N_QUBITS)

qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
compiled = transpile(qc, backend)
shots = 20000
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

# qc.measure(qubits, qubits) maps qubit i -> classical bit c_i, and Qiskit's
# count-dict keys print classical bits with c_{n-1} on the left and c_0 (the
# rightmost character) as the standard binary least-significant bit -- which
# is exactly qubit 0, matching the |k> convention used in apply_oracle (bit i
# of k lives on qubit i). So the key can be parsed as an ordinary binary
# number with no reversal.
def bitstring_to_int(bs: str) -> int:
    return int(bs, 2)

outcome_counts = Counter()
for bitstring, freq in counts.items():
    outcome_counts[bitstring_to_int(bitstring)] += freq

top_outcomes = [k for k, _ in outcome_counts.most_common(num_marked)]
top_outcomes_sorted = sorted(top_outcomes)

print(f"Grover iterations used: {iterations}")
print(f"Top {num_marked} measured outcomes (by frequency): {top_outcomes_sorted}")

quantum_matches_classical = top_outcomes_sorted == classical_solutions

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
    print(f"  classical: {classical_solutions}")
    print(f"  quantum  : {top_outcomes_sorted}")
