"""
Erdos problem #413 — quantum-testable instance.

Erdos problem #413 (see https://www.erdosproblems.com/413, tags:
"number theory", "iterated functions") concerns the cototient function
cot(x) = x - phi(x), where phi is Euler's totient. It is tied by the
erdosproblems.com dataset to OEIS sequence A005236 ("Noncototients: n such
that x - phi(x) = n has no solution").

Classical property tested here (finite, computable, and checked from first
principles in this script, not copied from OEIS):

    For the fixed target value n = 4 and the fixed search space
    x in {1, 2, ..., 15} (a 4-qubit index register), find all x with

        cot(x) = x - phi(x) = 4.

    Brute-force classical computation of phi(x) for x = 1..15 gives:
        cot(1..15) = [0, 1, 1, 2, 1, 4, 1, 4, 3, 6, 1, 8, 1, 8, 7]
    so the classical solution set for cot(x) = 4 is {6, 8} (exactly two
    marked items out of 16 basis states 0..15, since x=0 is never marked).

    n = 4 is itself *not* a noncototient (A005236 excludes it, since a
    solution exists: x=6 works, 6 - phi(6) = 6 - 2 = 4). This gives a
    genuine, checkable instance grounded in the same cot(x) = n equation
    that defines A005236, without fabricating a value: solvability of
    cot(x) = n for a *specific* small n is exactly the finite yes/no
    question A005236 is the sequence of "no" answers to.

Quantum approach: Grover's search algorithm on a 4-qubit index register
(16 basis states representing x = 0..15). The oracle is built directly from
the classically-computed marked set {6, 8} (a standard "oracle marking a
classically-known target subset" construction — this is a real amplitude-
amplification circuit, not a lookup pretending to be quantum). We run the
Grover-optimal number of iterations and confirm the simulator concentrates
measurement probability on the classically verified solutions {6, 8}.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def phi(n: int) -> int:
    """Euler's totient, computed from first principles (trial division)."""
    if n == 0:
        return 0
    result = n
    p = 2
    nn = n
    while p * p <= nn:
        if nn % p == 0:
            while nn % p == 0:
                nn //= p
            result -= result // p
        p += 1
    if nn > 1:
        result -= result // nn
    return result


def cot(n: int) -> int:
    """Cototient: x - phi(x)."""
    return n - phi(n)


def classical_solve(target: int, upper: int) -> list:
    """Brute-force all x in [1, upper) with cot(x) == target."""
    return [x for x in range(1, upper) if cot(x) == target]


# ---------------------------------------------------------------------------
# Classical ground truth
# ---------------------------------------------------------------------------

NUM_QUBITS = 4
N = 2 ** NUM_QUBITS  # 16 basis states, x = 0..15
TARGET_COTOTIENT = 4

cot_table = {x: cot(x) for x in range(1, N)}
solutions = classical_solve(TARGET_COTOTIENT, N)

print(f"Cototient table cot(x) for x=1..{N - 1}: {cot_table}")
print(f"Classical solutions of cot(x) = {TARGET_COTOTIENT} in [1,{N}): {solutions}")

assert solutions == [6, 8], f"unexpected classical solution set: {solutions}"

# ---------------------------------------------------------------------------
# Grover oracle marking exactly the classical solution set {6, 8}
# ---------------------------------------------------------------------------


def mark_state(qc: QuantumCircuit, value: int, num_qubits: int):
    """Flip the sign of basis state |value> using a multi-controlled Z,
    implemented via X-gates to remap value's 0-bits to 1-bits, an MCZ,
    then undoing the X-gates."""
    bits = [(value >> i) & 1 for i in range(num_qubits)]
    zero_positions = [i for i, b in enumerate(bits) if b == 0]

    for i in zero_positions:
        qc.x(i)

    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)

    for i in zero_positions:
        qc.x(i)


def build_oracle(marked_values: list, num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="oracle")
    for v in marked_values:
        mark_state(qc, v, num_qubits)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


num_solutions = len(solutions)
iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_solutions)))
print(f"Grover iterations: {iterations} (N={N}, marked={num_solutions})")

oracle = build_oracle(solutions, NUM_QUBITS)
diffuser = build_diffuser(NUM_QUBITS)

grover = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
grover.h(range(NUM_QUBITS))
for _ in range(iterations):
    grover.compose(oracle, inplace=True)
    grover.compose(diffuser, inplace=True)
grover.measure(range(NUM_QUBITS), range(NUM_QUBITS))

# ---------------------------------------------------------------------------
# Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(grover, backend)
shots = 4096
job = backend.run(transpiled, shots=shots)
counts = job.result().get_counts()

# Qiskit reports bitstrings MSB..LSB matching qubit order c[n-1]...c[0];
# convert each measured bitstring back to the integer x it represents.
measured_solution_shots = 0
total_shots = 0
for bitstring, count in counts.items():
    x = int(bitstring, 2)
    total_shots += count
    if x in solutions:
        measured_solution_shots += count

success_fraction = measured_solution_shots / total_shots
print(f"Measurement counts (by integer x): "
      f"{ {int(b, 2): c for b, c in counts.items()} }")
print(f"Fraction of shots landing on a classical solution {solutions}: "
      f"{success_fraction:.4f}")

# Grover's algorithm should concentrate the overwhelming majority of
# probability on the marked states after the optimal number of iterations.
THRESHOLD = 0.90
quantum_result_matches_classical = success_fraction >= THRESHOLD

if quantum_result_matches_classical:
    print("PASS")
else:
    print("FAIL")

assert quantum_result_matches_classical
