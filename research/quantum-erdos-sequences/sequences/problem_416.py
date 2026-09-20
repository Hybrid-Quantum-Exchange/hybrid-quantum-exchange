"""
Erdos problem #416 (erdosproblems.com), OEIS id used: A264810.

A264810 = "a(n) = number of m in [1, n] such that m = phi(k) for some k",
i.e. the running count of *totient values* (integers that occur as
Euler's-totient outputs). Equivalently, a(n) - a(n-1) is 1 exactly when n
is a totient value and 0 exactly when n is a "nontotient". This script
tests the underlying finite/computable property directly:

    Property tested: for fixed target value k = 6 and search domain
    m in {1, ..., N} with N = 16 (4 qubits of index space), find the set
    S(k) = { m in [1, N] : phi(m) = k }.

This is exactly the elementary fact A264810 is built from (whether each
integer k is hit by phi), just posed as a search problem instead of a
running count, which is what makes it Grover-searchable on a small
register.

Classical ground truth (computed here from first principles, trial
division for phi, no OEIS lookup):
    phi(1)=1  phi(2)=1  phi(3)=2  phi(4)=2  phi(5)=4  phi(6)=2  phi(7)=6
    phi(8)=4  phi(9)=6  phi(10)=4 phi(11)=10 phi(12)=4 phi(13)=12
    phi(14)=6 phi(15)=8 phi(16)=8
    => S(6) = {7, 9, 14}   (3 marked states out of 16, indices 6, 8, 13
       in 0-indexed 4-qubit basis states |m-1>)

Quantum approach: Grover's algorithm on a 4-qubit register representing
m-1 in {0,...,15}. The oracle marks the basis states whose index+1 is in
S(6) (computed classically above, not hand-copied from OEIS). With 3
marked items out of 16, the optimal number of Grover iterations is
floor(pi/4 * sqrt(16/3)) = 2. After running on the ideal AerSimulator,
the measurement distribution should be strongly concentrated on the 3
marked basis states {6, 8, 13} (i.e. bitstrings for m = 7, 9, 14).

PASS/FAIL: the script computes S(6) classically, builds and runs the
Grover circuit, and declares PASS iff the top-3 most frequent measured
outcomes (by shot count) are exactly the 3 classically-marked indices.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def phi(n: int) -> int:
    """Euler's totient function via trial division (first principles)."""
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


def classical_marked_indices(N: int, k: int):
    """Indices (0-indexed, index = m-1) of m in [1,N] with phi(m) == k."""
    return [m - 1 for m in range(1, N + 1) if phi(m) == k]


def build_oracle(n_qubits: int, marked_indices):
    """Phase-flip oracle marking each index in marked_indices."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        # Flip qubits that should be 0 so the marked state maps to |11..1>
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # Multi-controlled Z on the last qubit as target
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_indices, shots: int = 4096):
    N = 2 ** n_qubits
    n_marked = len(marked_indices)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_marked)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    N = 16          # search domain m in [1, N]
    k = 6            # target totient value: does phi(m) = 6 have solutions?
    n_qubits = int(math.log2(N))

    marked = classical_marked_indices(N, k)
    print(f"Classical: m in [1,{N}] with phi(m) = {k}: "
          f"{[i + 1 for i in marked]}  (0-indexed: {marked})")

    counts, iterations = run_grover(n_qubits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # counts keys are bitstrings 'q_{n-1}...q_0' (Qiskit convention, index 0
    # is the rightmost bit). Convert to integer index consistently with
    # build_oracle's little-endian convention (qubit i corresponds to bit i,
    # value = sum bit_i * 2^i).
    def bitstring_to_index(bs: str) -> int:
        # Qiskit prints classical bits as c[n-1] c[n-2] ... c[0]
        bits = bs[::-1]  # now bits[i] corresponds to qubit/classical-bit i
        return sum(int(bits[i]) * (2 ** i) for i in range(len(bits)))

    index_counts = {}
    for bitstring, c in counts.items():
        idx = bitstring_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + c

    ranked = sorted(index_counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = [idx for idx, _ in ranked[:len(marked)]]

    print("Measured index counts (top entries):",
          sorted(ranked, key=lambda kv: -kv[1])[:6])
    print(f"Top-{len(marked)} measured indices: {sorted(top_k)}  "
          f"vs classically marked: {sorted(marked)}")

    passed = sorted(top_k) == sorted(marked)

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
