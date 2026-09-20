"""
Erdos problem #259 -- quantum-testable instance.

Source: erdosproblems.com problem #259 (data/problems.yaml: number "259",
tags ["irrationality"], oeis: ["A371134"]).

OEIS A371134 is the decimal expansion of

    Sum_{k>=1, k squarefree} k / 2^k  =  Sum_{k>=1} k * mu(k)^2 / 2^k

(Erdos conjectured this constant irrational in 1981; Chen and Ruzsa proved
it irrational in 1999). The defining ingredient of the sequence is exactly
"k is squarefree", i.e. mu(k)^2 = 1, equivalently k is NOT divisible by any
perfect square p^2 for prime p.

Classical property tested here (finite, computable):
    For k in {0, 1, ..., 15} (4 bits), classify k as squarefree or not,
    where by convention 0 is treated as NOT squarefree (matching the
    summation range k >= 1 in A371134, and mu(0) = 0). Concretely:
        k is NOT squarefree  <=>  k == 0  or  4 | k  or  9 | k
    (within 0..15 the only square factors that can appear are 4 and 9,
    since 16 does not divide any k <= 15 other than via 4 already, and
    25 > 15).

The classically-computed "non-squarefree" set within 0..15 is
    MARKED = {0, 4, 8, 9, 12}
(this is computed in this script from first principles by trial division,
not copied from OEIS).

Quantum circuit: Grover's search over the 4-qubit computational basis
{0,...,15}, with an oracle that phase-flips exactly the states in MARKED
(built as a sequence of multi-controlled-Z gates, one per marked value,
each sandwiched between X gates that map that basis state to |1111>).
With N = 16 and M = |MARKED| = 5 marked states, one Grover iteration
(floor(pi/4 * sqrt(N/M)) = 1) maximizes the probability of measuring a
marked (non-squarefree) state.

Verification: run the circuit on the ideal AerSimulator, take the most
frequent measured outcomes, and check that (a) the single most-probable
outcome is a member of the classically-computed MARKED set, and (b) the
total measured probability mass landing on MARKED states is amplified
well above the naive M/N = 5/16 = 31.25% baseline (i.e. Grover actually
amplified the marked amplitudes). The script prints PASS iff both hold.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def is_squarefree(k: int) -> bool:
    """Classical, from-first-principles squarefree test via trial division."""
    if k <= 0:
        return False
    n = k
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        # strip out factors of p to keep checking remaining prime factors
        while n % p == 0:
            n //= p
        p += 1
    return True


def classical_marked_set():
    """Non-squarefree values in 0..N-1 (0 included by the k>=1 convention)."""
    return sorted(k for k in range(N) if not is_squarefree(k))


def bits_of(k: int, n_qubits: int):
    """Little-endian bit list (qubit 0 = LSB), matching Qiskit's ordering."""
    return [(k >> i) & 1 for i in range(n_qubits)]


def mark_state_phase_flip(qc: QuantumCircuit, k: int, n_qubits: int):
    """Phase-flip basis state |k> using X-sandwiched multi-controlled-Z."""
    bits = bits_of(k, n_qubits)
    flip_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in flip_qubits:
        qc.x(q)


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for k in marked:
        mark_state_phase_flip(qc, k, n_qubits)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc.decompose()


def main():
    marked = classical_marked_set()
    m = len(marked)
    print(f"Classical non-squarefree set in 0..{N-1}: {marked} (|M|={m})")
    assert marked == [0, 4, 8, 9, 12], "unexpected classical result"

    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / m))))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    shots = 8192
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints the classical bitstring with the highest-indexed bit on
    # the left and bit 0 (=qubit 0) on the right, so a plain int(...) parse
    # already gives k = sum(bit_i * 2**i), matching our bits_of() encoding.
    int_counts = {}
    for bitstring, c in counts.items():
        k = int(bitstring, 2)
        int_counts[k] = int_counts.get(k, 0) + c

    sorted_outcomes = sorted(int_counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (k: count):")
    for k, c in sorted_outcomes[:8]:
        flag = "marked" if k in marked else "unmarked"
        print(f"  k={k:2d} ({flag}): {c}")

    marked_mass = sum(c for k, c in int_counts.items() if k in marked) / shots
    baseline = m / N
    top_k, top_c = sorted_outcomes[0]

    print(f"Marked probability mass: {marked_mass:.4f} (naive baseline {baseline:.4f})")
    print(f"Most probable outcome: k={top_k}, in classical marked set: {top_k in marked}")

    ok_top = top_k in marked
    ok_amplified = marked_mass > baseline * 1.5  # clearly amplified above naive baseline

    if ok_top and ok_amplified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
