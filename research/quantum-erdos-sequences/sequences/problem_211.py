"""
Erdos problem #211 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "211"`).

Source metadata for problem 211:
    prize: $100
    status: proved (as of 2025-08-31)
    tags: ["geometry"]
    oeis: ["N/A"]

LIMITATION (reported honestly, per task instructions): problem 211 has no
OEIS sequence id attached in the source data (oeis: ["N/A"]). It is a
geometry statement (about point sets / convex position, per its tag), not a
number sequence, so there is no small finite "membership in the sequence"
or "early term" property of an OEIS sequence to test here. Fabricating an
OEIS-derived property for this problem would misrepresent the source data.

Best honest attempt instead: since no sequence-specific property exists to
verify, this script builds a genuine, self-contained, classically-checkable
quantum computation -- Grover's algorithm searching a small finite space for
primes -- and verifies the quantum result against a first-principles
classical computation. This is offered explicitly as a generic placeholder
quantum circuit, NOT as a demonstration of problem 211's mathematical
content, because problem 211 supplies no OEIS sequence to build one from.

Classical property actually tested:
    N = 16 (4 qubits, search space {0, ..., 15}).
    Target set S = { n in [0, 16) : n is prime }.
    Computed classically by trial division from first principles below.
    Grover's algorithm is run to amplify the marked (prime) states, and the
    most-frequently-measured basis states after the algorithm are compared
    against the classically computed prime set S.

PASS/FAIL: prints PASS if the set of basis states with the highest measured
probability equals the classical prime set S; FAIL otherwise.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_bits: int):
    N = 2 ** n_bits
    return sorted(x for x in range(N) if is_prime(x))


def build_oracle(n_bits: int, marked):
    """Phase-flip oracle marking each value in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked, shots: int = 4096):
    N = 2 ** n_bits
    M = len(marked)
    if M == 0 or M == N:
        raise ValueError("Grover requires 0 < M < N marked items")

    # Optimal number of Grover iterations for this N, M.
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_bits = 4  # N = 16
    N = 2 ** n_bits

    target_set = classical_primes(n_bits)
    print(f"Erdos problem 211: no OEIS id available (oeis: ['N/A']); "
          f"running generic Grover-search placeholder instead.")
    print(f"N = {N}, marked (classically computed primes in [0,{N})) = {target_set}")

    counts, iterations = run_grover(n_bits, target_set)
    print(f"Grover iterations used: {iterations}")

    shots = sum(counts.values())
    # Basis states sorted by measured probability, descending.
    sorted_states = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = len(target_set)
    top_states = sorted_states[:top_k]

    # Convert measured bitstrings (qiskit: c_{n-1}...c_0, little-endian to int)
    measured_top = sorted(int(bs, 2) for bs, _ in top_states)

    print("Top measured states (value: probability):")
    for bs, c in top_states:
        val = int(bs, 2)
        print(f"  {val:2d} ({bs}): {c / shots:.3f}")

    verified = (measured_top == target_set)

    print(f"Classical target set: {target_set}")
    print(f"Quantum top-{top_k} measured set: {measured_top}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
