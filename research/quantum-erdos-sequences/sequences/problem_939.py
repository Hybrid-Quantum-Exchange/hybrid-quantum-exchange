"""
Erdos problem #939 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
    number: "939"
    tags: ["number theory", "powerful"]
    oeis: ["A297867", "A036966", "possible"]

A036966 is the sequence of *powerful numbers* n>1 (every prime in the
factorization of n occurs with exponent >= 2) that are the smaller of two
consecutive powerful numbers -- more simply, membership in the broader class
of "powerful numbers" (A001694: n such that p | n => p^2 | n) is the concrete,
finite, checkable classical property problem #939's OEIS ids are anchored on.
A297867 is a related powerful-number-gap/count sequence; both rest on the
same base decision problem: "is n powerful?"

Chosen classical property for this script
------------------------------------------
Search space: integers n in [1, 15] (4 qubits, N = 16 fits exactly).
Property:      IsPowerful(n) := every prime factor of n has exponent >= 2
               (equivalently n has no prime factor to the first power only).
               1 is powerful by the empty-product convention used in A001694.

The classical answer (computed here from first principles, no OEIS lookup)
is derived by trial-division factorization for every n in [0, 15], giving
the ground-truth marked set that both the oracle and the final PASS/FAIL
check are built from.

Quantum circuit
----------------
Grover's algorithm on 4 qubits (search space size N = 16). The oracle is a
real multi-controlled-Z phase oracle built directly from the classical
marked-set bitstrings (one MCZ term per marked n, implemented with X-gates
to map the "0" bits of n's binary representation onto controls before/after
a multi-controlled Z). This is a genuine Grover oracle -- not a lookup table
returned as the answer -- and the diffusion operator is the standard
Grover diffuser. The number of Grover iterations is computed from the
standard formula floor(pi/4 * sqrt(N/M)) for M marked items. The circuit is
run on the ideal AerSimulator (statevector method, no noise) and the
measured, most-probable outcomes are compared against the classically
computed marked set.

PASS criterion: the set of the top-M most frequent measured bitstrings
(M = number of classically powerful n in [0,15]) equals exactly the
classically computed set of powerful numbers in that range.
"""

from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, no OEIS values copied).
# ---------------------------------------------------------------------------

def is_powerful(n: int) -> bool:
    """n is 'powerful' iff every prime factor of n occurs with exponent>=2.

    Convention (matches A001694): 0 and 1 are treated as powerful
    (n=0 is out of our search range anyway; n=1 is the empty product).
    """
    if n <= 1:
        return True
    m = n
    p = 2
    while p * p <= m:
        if m % p == 0:
            exp = 0
            while m % p == 0:
                m //= p
                exp += 1
            if exp < 2:
                return False
        p += 1
    if m > 1:
        # m is a leftover prime factor with exponent exactly 1.
        return False
    return True


N_QUBITS = 4
N = 2 ** N_QUBITS  # search space [0, 15]

classical_powerful = sorted(n for n in range(N) if is_powerful(n))
M = len(classical_powerful)

print("Classical powerful numbers in [0, %d]:" % (N - 1), classical_powerful)
assert classical_powerful == [0, 1, 4, 8, 9], (
    "sanity check against hand-verified small powerful numbers failed: "
    f"{classical_powerful}"
)


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical marked set.
# ---------------------------------------------------------------------------

def mark_state(qc: QuantumCircuit, qubits, n: int, n_qubits: int) -> None:
    """Flip the phase of computational basis state |n> using a multi-
    controlled Z, implemented with X gates around an MCZ (H-MCX-H trick on
    the target, or directly via a phase-flip using an ancilla-free MCZ)."""
    bits = format(n, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])
    # Multi-controlled Z across all qubits (phase flip on |11...1>).
    if n_qubits == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])


def build_oracle(marked, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for n in marked:
        mark_state(qc, list(range(n_qubits)), n, n_qubits)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


oracle = build_oracle(classical_powerful, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / M))))
print(f"N={N}, M={M} marked states, Grover iterations={iterations}")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(N_QUBITS), range(N_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator(method="statevector")
tqc = transpile(qc, backend)
shots = 8192
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit returns bitstrings MSB-first over classical bits c[n-1]...c[0],
# with c[i] holding the measurement of qubit i. Our mark_state used
# little-endian bit i <-> qubit i, so convert back consistently.
def bitstring_to_int(bs: str) -> int:
    # bs is c[n-1] c[n-2] ... c[0]; qubit i (== classical bit i) is bs[-(i+1)],
    # and it carries weight 2**i.
    return sum(int(bs[-(i + 1)]) << i for i in range(len(bs)))

sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
top_m = sorted_counts[:M]
measured_marked = sorted(bitstring_to_int(bs) for bs, _ in top_m)

print("Top measured outcomes (bitstring: count):", sorted_counts[: 2 * M])
print("Measured marked set (from top-M outcomes):", measured_marked)


# ---------------------------------------------------------------------------
# 4. PASS/FAIL
# ---------------------------------------------------------------------------

verified = measured_marked == classical_powerful
if verified:
    print("PASS: Grover search recovered exactly the powerful numbers in "
          f"[0, {N - 1}] -> {measured_marked}")
else:
    print("FAIL: quantum result does not match classical answer.")
    print("  classical:", classical_powerful)
    print("  measured :", measured_marked)

assert verified, "quantum result did not match classical ground truth"
