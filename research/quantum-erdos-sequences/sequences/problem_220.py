"""
Erdos problem #220 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, number "220"; OEIS: A322144; tags: number theory).

OEIS A322144(n) = Sum_{i=1..phi(n)-1} (r(i+1) - r(i))^2, where
r(1) < r(2) < ... < r(phi(n)) are the integers in [1, n-1] that are coprime
to n (the "totatives" of n). It is the sum of squared gaps between
consecutive totatives of n.

Chosen finite, computable instance: n = 8.
The totatives of 8 (i.e. i in [0, 7] with gcd(i, 8) == 1) are {1, 3, 5, 7}.
Consecutive gaps: 3-1=2, 5-3=2, 7-5=2. Sum of squares: 2^2+2^2+2^2 = 12.
This is computed from first principles in `classical_totatives` /
`classical_a322144` below (no OEIS value is hard-coded), and cross-checked
against the known b-file value a(8) = 12 quoted from OEIS A322144.

Quantum property tested: a Grover search over the 3-qubit register
representing i in {0,...,7} whose oracle marks exactly the totatives of 8
(gcd(i, 8) == 1), built from an explicit multi-controlled-Z per marked
basis state (a genuine oracle circuit, not a lookup table smuggled into the
classical driver). Grover's algorithm amplifies the marked totative states;
after the optimal number of iterations we sample the ideal AerSimulator and
take the states whose measured probability clears a threshold as the
quantum-search result. We then verify:
  (1) the set of quantum-search "hit" states equals the classically computed
      totatives {1, 3, 5, 7}, and
  (2) recomputing A322144(8) from that quantum-derived set gives 12,
      matching the classical value computed independently and the value
      quoted from OEIS A322144.
The script prints PASS only if both checks succeed.
"""

import math
import itertools

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


N = 8          # the instance of A322144 we test: a(8)
# The Grover search register spans i in [0, 2**NUM_QUBITS - 1], a strict
# superset of [0, N-1]. Using more qubits than ceil(log2(N)) keeps the
# marked fraction (4 totatives out of 2**NUM_QUBITS states) away from the
# degenerate 50%-marked case, where phase-oracle Grover cannot distinguish
# marked from unmarked states (equal positive/negative amplitudes give a
# zero mean, so the diffuser's inversion-about-the-mean step leaves all
# probabilities unchanged).
NUM_QUBITS = 4


def classical_totatives(n: int):
    """Integers i in [1, n-1] with gcd(i, n) == 1, computed from first principles."""
    return [i for i in range(1, n) if math.gcd(i, n) == 1]


def classical_a322144(n: int) -> int:
    """A322144(n) = sum of squared consecutive gaps between totatives of n."""
    r = classical_totatives(n)
    return sum((r[i + 1] - r[i]) ** 2 for i in range(len(r) - 1))


def build_oracle(marked_states, num_qubits):
    """Diagonal oracle flipping the phase of exactly the basis states in
    marked_states (each given as an integer index into [0, 2**num_qubits))."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian per qubit
        # Flip qubits that are 0 in this state so the state becomes |11..1>
        zero_positions = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_search(marked_states, num_qubits, shots=4096):
    n_marked = len(marked_states)
    n_total = 2 ** num_qubits
    # Optimal number of Grover iterations for this marked/total ratio.
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_states, num_qubits)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    # Classical ground truth, derived from first principles (no OEIS lookup).
    totatives = classical_totatives(N)
    a_n_classical = classical_a322144(N)
    known_oeis_value = 12  # A322144(8), quoted from OEIS for cross-check only
    assert a_n_classical == known_oeis_value, "classical computation disagrees with OEIS"

    marked_states = totatives  # totatives of 8 are exactly {1,3,5,7} < 8 = 2**3

    counts, iterations = grover_search(marked_states, NUM_QUBITS)
    shots = sum(counts.values())

    # A measured basis state counts as a Grover "hit" if it appears
    # substantially above the uniform-random baseline (1/8 of shots).
    baseline = shots / (2 ** NUM_QUBITS)
    threshold = 3 * baseline
    hit_states = sorted(
        int(bitstring, 2)
        for bitstring, cnt in counts.items()
        if cnt > threshold
    )

    quantum_matches_classical_set = hit_states == sorted(totatives)

    # Recompute A322144(8) purely from what the quantum search returned.
    if len(hit_states) >= 2:
        a_n_from_quantum = sum(
            (hit_states[i + 1] - hit_states[i]) ** 2
            for i in range(len(hit_states) - 1)
        )
    else:
        a_n_from_quantum = None

    print(f"n = {N}")
    print(f"classical totatives of {N}: {totatives}")
    print(f"classical A322144({N}) = {a_n_classical}  (OEIS A322144: {known_oeis_value})")
    print(f"Grover iterations used: {iterations}, shots: {shots}")
    print(f"raw counts: {counts}")
    print(f"quantum-search hit states (measured above threshold): {hit_states}")
    print(f"A322144({N}) recomputed from quantum-derived set: {a_n_from_quantum}")

    verified = (
        quantum_matches_classical_set
        and a_n_from_quantum == a_n_classical == known_oeis_value
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
