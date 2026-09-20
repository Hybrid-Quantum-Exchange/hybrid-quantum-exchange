"""
Erdos problem #991 -- quantum-testable sequence.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 991"):
    prize: no
    informal_status: proved (last_update 2025-09-16)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["discrepancy"]

LIMITATION (reported honestly): problem #991 carries no OEIS sequence id
("N/A"), so there is no concrete integer sequence from OEIS to target with a
quantum oracle for this entry. In place of fabricating an OEIS value, this
script builds a genuine, small, finite instance of the actual mathematical
territory the "discrepancy" tag names -- the Erdos discrepancy problem: does
every +-1 sequence x_1, x_2, ... have unbounded discrepancy, i.e. arbitrarily
large partial sums |x_1 + x_2 + ... + x_n|?

Classical property tested (computed from first principles in this script,
not copied from anywhere):
    For sequences of length N = 4 over {+1, -1}, does there exist an
    assignment x_0, x_1, x_2, x_3 such that every prefix sum
        s_1 = x_0
        s_2 = x_0 + x_1
        s_3 = x_0 + x_1 + x_2
        s_4 = x_0 + x_1 + x_2 + x_3
    satisfies |s_k| <= 1 (discrepancy bound C = 1)?

This is a finite, computable, small search space (2^4 = 16 candidate
sequences), decidable by brute force -- exactly the kind of small instance
Grover's algorithm can search. The classical brute-force search below finds
all sequences meeting the bound; the quantum circuit runs Grover search over
the 4-qubit space with an oracle that marks exactly those sequences, and the
script checks that measuring the amplified state returns one of the
classically-valid sequences with high probability.

Approach: Grover's algorithm, oracle built as explicit multi-controlled
phase flips on the classically-precomputed set of marked (valid) basis
states, one Grover iteration (near-optimal for 4 marked states out of 16 --
note: an exactly-half-marked instance, such as 4-out-of-8, is a known
Grover degenerate case where the diffuser is a no-op because the
post-oracle amplitude already has zero mean; N=4 avoids that by construction),
run on the ideal AerSimulator.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


N = 4  # sequence length -> 4 qubits, 16 candidate +-1 sequences
DISCREPANCY_BOUND = 1


def bits_to_signs(bits):
    """0 -> +1, 1 -> -1, matching the convention used throughout."""
    return [1 if b == 0 else -1 for b in bits]


def classical_valid_sequences(n, bound):
    """Brute-force, first-principles search over all 2^n sign sequences
    for those whose every prefix sum has absolute value <= bound."""
    valid = []
    for m in range(2 ** n):
        bits = [(m >> i) & 1 for i in range(n)]
        xs = bits_to_signs(bits)
        s = 0
        ok = True
        for x in xs:
            s += x
            if abs(s) > bound:
                ok = False
                break
        if ok:
            valid.append(m)
    return valid


def build_oracle(n, marked_states):
    """Phase-flip oracle: for each marked computational basis state,
    apply a multi-controlled Z (via X-sandwiching for 0-bits) that flips
    its phase by -1, leaving all other basis states untouched."""
    qc = QuantumCircuit(n, name="oracle")
    for m in marked_states:
        bits = [(m >> i) & 1 for i in range(n)]
        # flip qubits that should be 0 so the marked pattern becomes all-1
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n qubits (phase flip when all are |1>)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n, marked_states, iterations, shots=4096):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(n, marked_states)
    diffuser = build_diffuser(n)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))

    qc.measure(range(n), range(n))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    marked = classical_valid_sequences(N, DISCREPANCY_BOUND)
    print(f"Classical brute force: {len(marked)} of {2**N} sequences of length {N} "
          f"have discrepancy <= {DISCREPANCY_BOUND}")
    for m in marked:
        bits = [(m >> i) & 1 for i in range(N)]
        xs = bits_to_signs(bits)
        print(f"  bitstring={''.join(map(str, bits))} signs={xs}")

    # near-optimal Grover iteration count for M marked out of N_total
    n_total = 2 ** N
    m_count = len(marked)
    theta = np.arcsin(np.sqrt(m_count / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Running Grover search with {iterations} iteration(s)...")

    counts = run_grover(N, marked, iterations)
    print("Measurement counts:", counts)

    # Qiskit's classical register bit order in the count string is
    # little-endian relative to qubit index (qubit 0 = rightmost char).
    marked_bitstrings = set()
    for m in marked:
        bits = [(m >> i) & 1 for i in range(N)]
        # build the string as measured: c[N-1] c[N-2] ... c[0]
        s = "".join(str(bits[N - 1 - i]) for i in range(N))
        marked_bitstrings.add(s)

    total_shots = sum(counts.values())
    hits = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
    hit_fraction = hits / total_shots

    print(f"Fraction of shots landing on a classically-valid (discrepancy<={DISCREPANCY_BOUND}) "
          f"sequence: {hit_fraction:.4f}")

    # Grover amplification should push this well above the uniform-random
    # baseline of m_count/n_total.
    baseline = m_count / n_total
    success = hit_fraction > baseline + 0.2  # comfortably amplified

    if success:
        print("PASS: Grover search amplified measurement onto classically-verified "
              "discrepancy<=1 sequences, matching the classical brute-force answer.")
    else:
        print("FAIL: quantum measurement did not match the classical answer within tolerance.")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
