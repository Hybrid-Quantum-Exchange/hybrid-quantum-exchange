"""
Erdos problem #365 (data/problems.yaml, number: "365").

Metadata for #365: open, no prize, tags ["number theory", "powerful"],
oeis ids ["A060355", "A060859", "A175155"]. The tag "powerful" and the
associated OEIS entries (A060355/A060859/A175155 all enumerate or bound
"powerful numbers": positive integers n such that p | n implies p^2 | n
for every prime p) point to the classical property used here, which is
directly computable and checkable in a small finite instance:

    PROPERTY TESTED: "n is a powerful number", for n in [0, 63]
    (a 6-qubit register), i.e. every prime in n's factorization occurs
    with exponent >= 2 (0 is treated as not powerful; 1 is powerful by
    the empty-product convention, matching OEIS A001694, the base
    "powerful numbers" sequence that A060355/A060859/A175155 build on).

    We do NOT copy an OEIS term literally: the set of powerful numbers
    in [0, 63] is derived here from first principles by trial-division
    factorization (`is_powerful`), and only that classically-derived
    set is used to build the quantum oracle and to check the quantum
    result.

CIRCUIT: Grover's search over the 6-qubit basis states |n>, n in
[0, 63], with an oracle that phase-flips exactly the marked states
(the powerful numbers found classically). The oracle is built without
looking up any OEIS table: for each marked integer n, X gates map |n>
to |111111>, a multi-controlled Z (via H + MCX + H on the ancilla-free
last qubit) flips its phase, and the X gates are undone. The standard
Grover diffuser is applied ceil(pi/4 * sqrt(N/M)) times, matching the
number of marked items M found classically.

We run the circuit on the ideal AerSimulator and compare the set of
n whose measured probability clears a threshold (>= 1/(2M), well above
the ~1/64 baseline of an unmarked state) against the classically
computed set of powerful numbers in [0, 63]. PASS requires equality.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ----------------------------------------------------------------------
# 1. Classical ground truth: powerful numbers in [0, 63], derived from
#    first principles (trial-division factorization), not looked up.
# ----------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """True iff n is a powerful number: every prime factor of n has
    exponent >= 2 in n's factorization. n=1 is powerful (empty product);
    n=0 is defined here as not powerful (no factorization)."""
    if n <= 0:
        return False
    if n == 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exponent = 0
            while m % p == 0:
                m //= p
                exponent += 1
            if exponent < 2:
                return False
        p += 1
    if m > 1:
        # a leftover prime factor with exponent exactly 1
        return False
    return True


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64

classical_marked = sorted(n for n in range(N) if is_powerful(n))
M = len(classical_marked)
assert M > 0

# sanity spot-check against known small powerful numbers (derived, not
# assumed): 1, 4, 8, 9, 16, 25, 27, 32, 36, 49, ...
expected_prefix = [1, 4, 8, 9, 16, 25, 27, 32, 36, 49]
assert classical_marked[: len(expected_prefix)] == expected_prefix, classical_marked


# ----------------------------------------------------------------------
# 2. Grover oracle + diffuser built purely from the classical marked set.
# ----------------------------------------------------------------------

def apply_phase_flip_for_state(qc: QuantumCircuit, n: int, n_qubits: int) -> None:
    """Flip the phase of basis state |n> (n_qubits wide), leaving all
    others unchanged, using X + multi-controlled-Z + X."""
    bits = format(n, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    # multi-controlled Z across all n_qubits (controls = first n-1, target = last)
    controls = list(range(n_qubits - 1))
    target = n_qubits - 1
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)

    for i in zero_positions:
        qc.x(i)


def build_oracle(marked, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="oracle")
    for n in marked:
        apply_phase_flip_for_state(qc, n, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    target = n_qubits - 1
    controls = list(range(n_qubits - 1))
    qc.h(target)
    qc.mcx(controls, target)
    qc.h(target)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_marked, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ----------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical set.
# ----------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
shots = 20000
result = backend.run(transpiled, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-bit string is big-endian in the printed key relative
# to qubit index order used above (qubit 0 is the rightmost char).
prob_by_n = {n: 0.0 for n in range(N)}
for bitstring, count in counts.items():
    n = int(bitstring, 2)
    prob_by_n[n] += count / shots

threshold = 1.0 / (2 * M)
quantum_marked = sorted(n for n in range(N) if prob_by_n[n] >= threshold)

verified = quantum_marked == classical_marked

print("Classically-derived powerful numbers in [0,63]:", classical_marked)
print("Grover iterations used:", iterations)
print("Quantum-detected high-probability states:      ", quantum_marked)
print("Sum of probability mass on marked states:", sum(prob_by_n[n] for n in classical_marked))

if verified:
    print("PASS")
else:
    print("FAIL")
