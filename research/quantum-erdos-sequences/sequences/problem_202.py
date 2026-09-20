"""
Erdos problem #202 -- quantum-testable instance
=================================================

Erdos problem #202 (erdosproblems.com, tag: "covering systems") is solved
(Lean-formalized, 2026-05-14) and is linked in the problems database to
OEIS sequence A389975:

    A389975: "Maximum cardinality of a set of disjoint congruence classes
    with distinct moduli, each at most n."
    a(1..20) = 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6

    A389975's own published example: for n = 6, a(6) = 2, witnessed by the
    two disjoint congruence classes {0 (mod 2), 1 (mod 4)} -- "disjoint"
    meaning no integer x satisfies both x == 0 (mod 2) and x == 1 (mod 4)
    simultaneously.

Classical property tested (finite, computable)
------------------------------------------------
Fix two moduli both <= n = 6: m1 = 2, m2 = 6 (6 is the largest modulus
allowed at n = 6, and 2 is the modulus of the OEIS witness class itself).
Two congruence classes (r1 mod m1) and (r2 mod m2) are DISJOINT (share no
integer) iff, by CRT reasoning on g = gcd(m1, m2):

    r1 mod g != r2 mod g          (g = gcd(2, 6) = 2)

There are 2 * 6 = 12 possible (r1, r2) pairs with r1 in {0,1}, r2 in
{0,...,5}. Exactly 6 of them are disjoint pairs (this is checked directly
by brute-force CRT/search below, in Python, before any quantum code runs):

    disjoint pairs: (0,1) (0,3) (0,5) (1,0) (1,2) (1,4)

This matches A389975's claim that a disjoint pair with distinct moduli
<= 6 exists (a(6) = 2 >= 2): (0 mod 2, 1 mod 6) is such a pair, and
generalizes the sequence's own published witness (0 mod 2, 1 mod 4).

Quantum circuit
----------------
A 4-qubit Grover search over r1 (1 qubit, values 0-1) and r2 (3 qubits,
values 0-7, of which only 0-5 are valid residues mod 6; values 6-7 are
simply never marked). N = 16 basis states, M = 6 of them marked. The
oracle is a diagonal phase-flip built directly from the classically
computed set of disjoint (marked) pairs above (a legitimate Grover oracle
constructed from a known marked set, exactly as in the standard
"diagonal oracle" formulation of Grover's algorithm). A diffusion
(inversion-about-the-mean) operator follows. Because M/N is not close to
1/2 here, Grover amplification is genuinely useful: the number of
iterations is chosen (from a handful of statevector trials, all still
classical/deterministic simulation of the *same* circuit family) to
maximize the total measured probability mass on the 6 marked states.

The circuit is run on the ideal AerSimulator. PASS requires: (a) the
brute-force classical check independently confirms the marked set has
size 6 and contains the generalized witness (0,1); and (b) sampling the
quantum circuit yields measured outcomes whose (r1, r2) decoding is
overwhelmingly (>95% of shots) one of the 6 classically-disjoint pairs,
i.e. Grover amplifies onto the classically verified marked states far
above the 6/16 = 37.5% baseline of uniform sampling.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import DiagonalGate


def is_disjoint(r1, m1, r2, m2):
    """Two classes (r1 mod m1), (r2 mod m2) are disjoint iff no integer
    satisfies both. Check directly over one full period (lcm(m1, m2))."""
    L = math.lcm(m1, m2)
    for x in range(L):
        if x % m1 == r1 and x % m2 == r2:
            return False
    return True


def build_grover_circuit(n_qubits, marked_indices, iterations):
    N = 2 ** n_qubits
    diag = [1.0] * N
    for idx in marked_indices:
        diag[idx] = -1.0
    oracle = DiagonalGate(diag)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    # Diffusion operator (inversion about the mean)
    def diffuser():
        d = QuantumCircuit(n_qubits, name="diffuser")
        d.h(range(n_qubits))
        d.x(range(n_qubits))
        d.h(n_qubits - 1)
        d.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        d.h(n_qubits - 1)
        d.x(range(n_qubits))
        d.h(range(n_qubits))
        return d.to_gate()

    diff_gate = diffuser()

    for _ in range(iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diff_gate, range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def classical_solve_padded(m1, m2, r2_bits):
    """Like classical_solve, but r2's index uses r2_bits bits (m2 <= 2**r2_bits),
    so indices for r2 >= m2 simply never appear (never marked)."""
    marked = []
    for r1 in range(m1):
        for r2 in range(m2):
            if is_disjoint(r1, m1, r2, m2):
                idx = r1 * (2 ** r2_bits) + r2
                marked.append((idx, r1, r2))
    return marked


def best_iterations(n_qubits, marked_indices, max_try=6):
    """Try a few iteration counts and keep the one that maximizes the
    statevector probability mass on the marked indices (still a
    deterministic classical simulation of the exact same circuit)."""
    backend = AerSimulator(method="statevector")
    best_it, best_p = 0, -1.0
    for it in range(0, max_try + 1):
        qc = build_grover_circuit(n_qubits, marked_indices, it)
        qc.remove_final_measurements()
        qc.save_statevector()
        tqc = transpile(qc, backend)
        sv = backend.run(tqc).result().get_statevector()
        probs = np.abs(np.asarray(sv)) ** 2
        p = sum(probs[i] for i in marked_indices)
        if p > best_p:
            best_p, best_it = p, it
    return best_it, best_p


def main():
    m1, m2 = 2, 6  # moduli both <= n = 6, matching A389975's a(6)=2 case
    r1_bits, r2_bits = 1, 3
    n_qubits = r1_bits + r2_bits  # = 4
    N = 2 ** n_qubits

    # --- classical, first-principles ---
    marked = classical_solve_padded(m1, m2, r2_bits)
    marked_indices = sorted(idx for idx, r1, r2 in marked)
    marked_pairs = sorted((r1, r2) for idx, r1, r2 in marked)
    M = len(marked_indices)

    witness = (0, 1)  # generalizes OEIS A389975's a(6)=2 witness (0 mod 2, 1 mod 4)
    assert witness in marked_pairs, "classical check disagrees with OEIS witness"
    assert M == 6, f"expected 6 disjoint pairs out of 12, got {M}"

    print(f"Classical brute force: moduli m1={m1}, m2={m2}")
    print(f"Disjoint (marked) (r1,r2) pairs: {marked_pairs}")
    print(f"OEIS A389975 a(6)=2 witness family (0 mod 2, 1 mod 6) present: "
          f"{witness in marked_pairs}")

    # --- quantum ---
    iterations, predicted_p = best_iterations(n_qubits, marked_indices)
    print(f"Chosen Grover iterations: {iterations} "
          f"(predicted marked probability {predicted_p:.4f})")
    qc = build_grover_circuit(n_qubits, marked_indices, iterations)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    def decode(bitstring):
        # qiskit classical-register string is c(n-1)...c1c0; reverse so
        # bits[i] corresponds to qubit i. The basis index used to build
        # the diagonal oracle was idx = r1 * 2**r2_bits + r2, i.e. qubits
        # 0..r2_bits-1 (LSB) hold r2 and qubit r2_bits holds r1.
        bits = bitstring[::-1]
        r2 = int(bits[0]) | (int(bits[1]) << 1) | (int(bits[2]) << 2)
        r1 = int(bits[3])
        return r1, r2

    total_marked_shots = 0
    for bitstring, count in counts.items():
        r1, r2 = decode(bitstring)
        if (r1, r2) in marked_pairs:
            total_marked_shots += count

    marked_fraction = total_marked_shots / shots
    baseline = M / N
    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts (raw): {counts}")
    print(f"Fraction of shots decoding to a classically-disjoint pair: "
          f"{marked_fraction:.4f}  (uniform-sampling baseline: {baseline:.4f})")

    # PASS requires strong amplification onto the classically-verified
    # marked set, well above the uniform-sampling baseline of M/N.
    verified = marked_fraction > 0.95 and marked_fraction > 2 * baseline

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
