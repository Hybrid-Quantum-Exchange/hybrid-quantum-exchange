"""
Erdos problem #692 (see erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '692'"): status is "disproved (Lean)", prize "no", and critically its
OEIS field is `["N/A"]` -- no OEIS sequence id is associated with this problem. Its
tags are ["number theory", "divisors"].

LIMITATION, stated up front: because problem #692 has no OEIS sequence id, there is
no specific "quantum-testable sequence" of the kind the other lanes in this library
build (e.g. "is integer k a term of OEIS Axxxxxx"). Fabricating an OEIS id or a
literal sequence value for this entry would misrepresent the source data. Instead,
in the spirit of the problem's own tags (number theory / divisors), this script
builds a REAL, genuinely computed, small finite divisor-counting search problem and
solves it with Grover's algorithm on the ideal AerSimulator:

    Classical property tested:
        S = { n in [0, 63] : n has exactly 3 positive divisors }
    (equivalently: n = p^2 for a prime p -- the standard number-theoretic
    characterization of tau(n) = 3, since only 1, p, p^2 divide p^2). This keeps
    the marked set small relative to the 64-item search space, which is what makes
    Grover's quadratic speedup meaningful here. It is computed from first
    principles below with a plain trial-division divisor counter -- no OEIS value
    is copied.

Quantum circuit:
    A 6-qubit register indexes n in [0, 63]. The classically-precomputed marked set
    S is used to build a Grover oracle (X-gates + multi-controlled phase flip, one
    per marked basis state) that flags exactly the divisor-count-4 integers. The
    standard Grover diffusion operator is applied for the optimal number of
    iterations. Running the circuit on AerSimulator should concentrate measurement
    probability onto the states in S.

Verification: the script computes S classically, runs the Grover circuit, takes the
most-frequent measured outcomes (as many as |S|), and checks they are exactly S.
Prints PASS or FAIL.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def divisor_count(n: int) -> int:
    """Count positive divisors of n via trial division (first principles)."""
    if n == 0:
        return 0
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count


def classical_marked_set():
    return sorted(n for n in range(N) if divisor_count(n) == 3)


def bits_of(n: int, width: int):
    """Little-endian bit list (qubit 0 = LSB), matching Qiskit's bit ordering."""
    return [(n >> i) & 1 for i in range(width)]


def add_oracle_mark(qc: QuantumCircuit, n: int, qubits):
    """Flip the phase of basis state |n> using X-gates + multi-controlled Z."""
    bits = bits_of(n, len(qubits))
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def build_grover_circuit(marked, n_qubits, iterations):
    qubits = list(range(n_qubits))
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(qubits)

    for _ in range(iterations):
        # Oracle: phase-flip each marked basis state
        for m in marked:
            add_oracle_mark(qc, m, qubits)

        # Diffusion operator (inversion about the mean)
        qc.h(qubits)
        qc.x(qubits)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        qc.x(qubits)
        qc.h(qubits)

    qc.measure(qubits, qubits)
    return qc


def main():
    marked = classical_marked_set()
    print(f"Classical property: n in [0,{N-1}] with exactly 3 positive divisors (n = p^2)")
    print(f"Classical marked set S (|S|={len(marked)}): {marked}")

    # Sanity-check a few known values by hand (first-principles cross-check)
    assert divisor_count(4) == 3   # divisors 1,2,4  (2^2)
    assert divisor_count(9) == 3   # divisors 1,3,9  (3^2)
    assert divisor_count(25) == 3  # divisors 1,5,25 (5^2)
    assert divisor_count(6) == 4   # not in S

    M = len(marked)
    optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))
    print(f"Grover iterations used: {optimal_iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, optimal_iterations)

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # qc.measure(qubits, qubits) maps qubit i -> classical bit i, and Qiskit's
    # counts key is the classical register read as bits c[n-1]...c[0]. Since our
    # basis-state index for integer n is n itself (Statevector/Aer both index
    # basis states by the integer value of qubit0=LSB .. qubit(n-1)=MSB, which is
    # exactly standard binary), the bitstring is the plain binary encoding of n --
    # no reversal needed (verified empirically against build_grover_circuit output).
    def bitstring_to_int(bs: str) -> int:
        return int(bs, 2)

    outcome_counts = {}
    for bitstring, c in counts.items():
        val = bitstring_to_int(bitstring)
        outcome_counts[val] = outcome_counts.get(val, 0) + c

    ranked = sorted(outcome_counts.items(), key=lambda kv: -kv[1])
    top_k = sorted(v for v, _ in ranked[:M])

    print(f"Top-{M} most frequent measured integers: {top_k}")

    marked_prob = sum(outcome_counts.get(v, 0) for v in marked) / shots
    print(f"Total measured probability landing on S: {marked_prob:.3f}")

    verified = (top_k == marked) and (marked_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
