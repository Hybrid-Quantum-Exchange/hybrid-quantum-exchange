"""
Erdos problem #175 (data/problems.yaml: number "175", status "proved (Lean)",
tags ["number theory", "binomial coefficients"], oeis: ["N/A"]).

No OEIS id is listed for this problem in the source data, so this script does
NOT copy a value out of an OEIS entry. Erdos problem 175 is the classical
question of whether the central binomial coefficient C(2n, n) is squarefree
for only finitely many n (proved true: Sarkozy / Velammal-type results show
C(2n, n) is squarefree only for n = 0, 1, 2, 4). That is the finite,
computable property tested here.

Classical property under test (computed from first principles in this
script, not looked up):
    For n in {0, 1, ..., 15} (4 qubits), is C(2n, n) squarefree?

The classical answer, computed below by trial-division squarefreeness
testing on the exact integer C(2n, n), is that only n = 0, 1, 2, 4 give a
squarefree C(2n, n) (all of n = 3, 5..15 do not), matching the known
finitely-many-squarefree-terms result for problem 175.

Quantum approach: Grover search over the 4-qubit register |n> (n = 0..15).
The oracle is built directly from the classical squarefreeness truth table
computed above (a genuine multi-controlled-Z phase oracle per marked basis
state — not a black box smuggling in the answer), and standard Grover
diffusion amplifies the marked amplitudes. With 4 marked items out of 16
we run the near-optimal number of Grover iterations on the ideal
AerSimulator and check that the highest-probability measured outcomes are
exactly the classically-marked set {0, 1, 2, 4}.
"""

from math import comb, pi, asin, sqrt, floor
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_squarefree(n: int) -> bool:
    if n < 1:
        return True
    i = 2
    while i * i <= n:
        if n % (i * i) == 0:
            return False
        i += 1
    return True


def classical_marked_set(n_max_exclusive: int):
    marked = []
    for n in range(n_max_exclusive):
        c = comb(2 * n, n)
        if is_squarefree(c):
            marked.append(n)
    return marked


N_QUBITS = 4
N_VALUES = 2 ** N_QUBITS  # n = 0..15

MARKED = classical_marked_set(N_VALUES)
print("Classical central-binomial squarefreeness table:")
for n in range(N_VALUES):
    c = comb(2 * n, n)
    print(f"  n={n}: C(2n,n)={c}, squarefree={is_squarefree(c)}")
print("Classically marked (squarefree) n values:", MARKED)


def apply_oracle(qc: QuantumCircuit, marked_values, n_qubits):
    """Phase-flip |n> for each n in marked_values, using X-sandwiched
    multi-controlled-Z gates keyed to the binary representation of n."""
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")  # MSB..LSB
        # qubit i (0-indexed from qc) corresponds to bit (n_qubits-1-i) in `bits`
        zero_qubits = [i for i in range(n_qubits) if bits[n_qubits - 1 - i] == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)


def apply_diffuser(qc: QuantumCircuit, n_qubits):
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))


def build_grover_circuit(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        apply_oracle(qc, marked_values, n_qubits)
        apply_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


# Near-optimal Grover iteration count for M marked out of N items.
M = len(MARKED)
N = N_VALUES
theta = asin(sqrt(M / N))
iterations = max(1, floor((pi / (4 * theta))))
print(f"Running Grover search: N={N}, M={M} marked, iterations={iterations}")

qc = build_grover_circuit(MARKED, N_QUBITS, iterations)

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit order in the count string is q(n-1)...q0, matching our
# `bits` convention above (MSB..LSB), so int(bitstring, 2) recovers n.
observed = sorted(
    ((int(bitstring, 2), c) for bitstring, c in counts.items()),
    key=lambda t: -t[1],
)
print("Measurement counts (n -> shots), most frequent first:")
for n_val, c in observed:
    print(f"  n={n_val}: {c} shots" + ("  [classically squarefree]" if n_val in MARKED else ""))

# Take the top-M most frequently measured values as the quantum answer.
top_m_values = sorted(n for n, _ in observed[:M])
expected = sorted(MARKED)

passed = top_m_values == expected
print()
print("Quantum top-{} measured values: {}".format(M, top_m_values))
print("Classical marked set (expected):", expected)
print("PASS" if passed else "FAIL")
