"""
Erdos problem #394 (erdosproblems.com/394) -- quantum-testable instance.

OEIS id used: A344005
  a(n) = smallest positive integer m such that n divides the oblong
  (pronic) number m*(m+1).

Classical property tested here (small finite instance, n = 15):
  Let S(n) = { m in [1, 15] : n | m*(m+1) }  (search space restricted to
  4-bit integers m in [1,15], i.e. the nonzero states of a 4-qubit register).
  For n = 15 this is computed from first principles below by brute force:
      S(15) = { m in [1,15] : 15 | m*(m+1) }
  and OEIS A344005(15) = min(S(15)) is the quantity the sequence records.

  This script:
    1. Computes S(15) classically by brute force over m = 1..15 (no OEIS
       value is copied -- it is derived here and only afterwards compared
       to the known OEIS term as a sanity check).
    2. Builds a genuine Grover search circuit over the 4-qubit register
       representing m in [0,15]. The oracle is constructed purely from
       the classically-computed set S(15): it phase-flips exactly the
       basis states whose bit pattern equals an element of S(15) (using
       X-gate sandwiched multi-controlled-Z gates, the standard Grover
       oracle-from-marked-set construction). This is a real amplitude
       amplification circuit -- diffuser included -- with the optimal
       number of Grover iterations for |S(15)|/16 marked density.
    3. Runs the circuit on the ideal AerSimulator, and checks that Grover
       amplification concentrated the measurement probability onto states
       of the true marked set S(15) (i.e. the quantum search finds a
       genuine solution of "15 | m*(m+1)" with high probability), and in
       particular that the single most-frequently measured outcome is a
       correct member of S(15). It further explicitly reports whether the
       true minimum (the OEIS A344005(15) value) was recovered among the
       measured Grover hits, since Grover amplifies all marked states
       uniformly and is not itself a "find the minimum" algorithm -- the
       minimum is identified by classical post-processing of the quantum
       search's output, as is standard.

  PASS criterion: (a) S(15) computed classically is non-empty and matches
  OEIS A344005(15) = 5 as its minimum, and (b) the quantum circuit's most
  frequent measured outcome is a genuine member of S(15) (Grover search
  succeeded), and the minimum element of S(15) appears among the measured
  outcomes with non-negligible probability.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N_STATES = 2 ** N_QUBITS  # 16
N_VALUE = 15  # the n in OEIS A344005(n)


def classical_marked_set(n, n_qubits):
    """Brute-force S(n) = {m in [1, 2^n_qubits - 1] : n | m*(m+1)}."""
    limit = 2 ** n_qubits
    return sorted(m for m in range(1, limit) if (m * (m + 1)) % n == 0)


def bits_of(m, n_qubits):
    """Little-endian bit list of m (bit 0 = qubit 0), length n_qubits."""
    return [(m >> i) & 1 for i in range(n_qubits)]


def add_mark_phase_flip(qc, m, n_qubits):
    """Phase-flip exactly basis state |m> using X-sandwiched MCZ."""
    bits = bits_of(m, n_qubits)
    zero_qubits = [i for i, b in enumerate(bits) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    # multi-controlled Z on all n_qubits (phase flip |11...1>)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        add_mark_phase_flip(qc, m, n_qubits)
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
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    # Step 1: classical brute force, from first principles.
    marked = classical_marked_set(N_VALUE, N_QUBITS)
    print(f"n = {N_VALUE}, search space m in [1, {N_STATES - 1}]")
    print(f"Classically computed S({N_VALUE}) = {marked}")

    if not marked:
        print("No marked states in this small instance -- cannot build a "
              "nontrivial Grover search. FAIL")
        return False, False

    classical_min = min(marked)
    print(f"Classical min(S({N_VALUE})) = {classical_min} "
          f"(expected to equal OEIS A344005({N_VALUE}) = 5)")
    matches_oeis = (classical_min == 5)
    print(f"Matches known OEIS A344005({N_VALUE}) = 5: {matches_oeis}")

    # Step 2: build and run the Grover circuit.
    m_count = len(marked)
    theta = math.asin(math.sqrt(m_count / N_STATES))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"|S| = {m_count}, optimal Grover iterations = {iterations}")

    qc = build_grover_circuit(marked, N_QUBITS, iterations)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: classical register bit string is big-endian in the
    # printed key (c[n-1] ... c[0]); convert back to integer m.
    int_counts = Counter()
    for bitstring, c in counts.items():
        m = int(bitstring, 2)
        int_counts[m] += c

    top_outcomes = int_counts.most_common(5)
    print(f"Top measured outcomes (m: count): {top_outcomes}")

    marked_set = set(marked)
    total_marked_shots = sum(c for m, c in int_counts.items() if m in marked_set)
    marked_fraction = total_marked_shots / shots
    print(f"Fraction of shots landing on a genuine member of S({N_VALUE}): "
          f"{marked_fraction:.3f}")

    top_m, top_c = top_outcomes[0]
    top_is_marked = top_m in marked_set
    print(f"Most frequent measured outcome m={top_m} is a genuine "
          f"solution of {N_VALUE} | m*(m+1): {top_is_marked}")

    min_hits = int_counts.get(classical_min, 0)
    min_recovered = min_hits > 0
    print(f"Classical minimum m={classical_min} observed among quantum "
          f"measurements: {min_recovered} ({min_hits}/{shots} shots)")

    ran_ok = True
    verified = bool(
        matches_oeis
        and top_is_marked
        and marked_fraction > 0.8
        and min_recovered
    )

    print("PASS" if verified else "FAIL")
    return ran_ok, verified


if __name__ == "__main__":
    ok, verified = main()
    if not verified:
        raise SystemExit(1)
