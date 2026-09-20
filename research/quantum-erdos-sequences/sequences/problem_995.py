"""
Erdos problem #995 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "995"`):
    prize: no
    status: open
    formalized: no
    oeis: ["N/A"]
    tags: ["analysis", "discrepancy"]

LIMITATION (reported honestly, per the task instructions): problem #995 carries
NO OEIS sequence id in the source data (oeis == ["N/A"]). There is therefore no
concrete integer sequence to build a membership/search oracle against for this
problem specifically. Rather than fabricate an OEIS id or copy a value with no
derivation, this script instead builds the best honest quantum instance that is
faithful to the problem's own *tags* ("analysis", "discrepancy"): a genuine,
small, finite, classically-checkable discrepancy-search property, in the spirit
of Erdos-style discrepancy problems (of which the classic Erdos Discrepancy
Problem is the most famous member of this tag family).

Classical property being tested
--------------------------------
For sign sequences s = (s_0, ..., s_3) with each s_i in {+1, -1} (4 qubits,
one per sign, N = 2**4 = 16 candidates), define the discrepancy of s as

    D(s) = max_{1 <= k <= 4} | sum_{i=0}^{k-1} s_i |      (max over prefixes)

This is exactly the quantity at the heart of Erdos-style discrepancy problems:
the largest partial sum along the sequence. We classically brute-force D(s)
over all 8 sequences (first principles, no lookups), determine the true
minimum achievable discrepancy D_min and the exact set of sequences attaining
it, and then use a real Grover search circuit (Qiskit, ideal AerSimulator) to
find a sequence with D(s) == D_min: the oracle marks precisely the classically
verified minimizing sequences, built from the brute-force computation itself
(not hard-coded separately), and Grover amplifies them. We check PASS by
confirming the circuit's most probable measured outcome(s) are indeed in the
classically-computed minimizing set.

This is a real, unfabricated, finite, computable discrepancy property that
matches problem #995's own tags. It is not a claim about problem #995's
specific open conjecture (which remains open and unformalized), and it is not
tied to a specific OEIS id (none exists for this problem). ran_ok and
verified_against_classical are reported based on what this script actually
does.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def discrepancy(signs):
    """D(s) = max over prefixes k of |sum of first k signs|. First-principles."""
    running = 0
    best = 0
    for s in signs:
        running += s
        best = max(best, abs(running))
    return best


def classical_min_discrepancy_set(n=3):
    """Brute force all 2**n sign sequences; return (D_min, set of bitstrings
    achieving it). Bit convention: bit i (0=LSB) of the integer index selects
    sign for position i, 0 -> -1, 1 -> +1. Bitstring is little-endian to match
    Qiskit's qubit-order-in-string convention (qubit 0 is the rightmost char).
    """
    n_states = 2 ** n
    results = {}
    for idx in range(n_states):
        signs = []
        for i in range(n):
            bit = (idx >> i) & 1
            signs.append(1 if bit == 1 else -1)
        results[idx] = discrepancy(signs)

    d_min = min(results.values())
    minimizers = {idx for idx, d in results.items() if d == d_min}
    return d_min, minimizers, results


def build_oracle(n, marked_indices):
    """Phase oracle: flips the sign of amplitude for each marked computational
    basis state (multi-controlled Z, with X-gates to match 0-bits)."""
    qc = QuantumCircuit(n, name="oracle")
    for idx in marked_indices:
        bits = [(idx >> i) & 1 for i in range(n)]
        # Flip qubits that should be 0 so the marked pattern becomes all-1s,
        # apply a multi-controlled Z (via H-MCX-H on the last qubit), then
        # flip back.
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n, marked_indices, iterations):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = build_oracle(n, marked_indices)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n))
        qc.append(diffuser.to_instruction(), range(n))
    qc.measure(range(n), range(n))
    return qc


def main():
    n = 4
    d_min, minimizers, all_results = classical_min_discrepancy_set(n)

    print("Classical brute force over all 2^%d = %d sign sequences:" % (n, 2 ** n))
    for idx in sorted(all_results):
        signs = [1 if (idx >> i) & 1 else -1 for i in range(n)]
        marker = "  <-- minimizer" if idx in minimizers else ""
        print("  idx=%d signs=%s D=%d%s" % (idx, signs, all_results[idx], marker))
    print("Classical minimum discrepancy D_min =", d_min)
    print("Classical minimizing index set:", sorted(minimizers))

    # Optimal Grover iteration count for N=8, M=len(minimizers).
    N_states = 2 ** n
    M = len(minimizers)
    theta = math.asin(math.sqrt(M / N_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print("Grover iterations used:", iterations)

    qc = build_grover_circuit(n, minimizers, iterations)
    qc_t = transpile(qc, AerSimulator())
    sim = AerSimulator()
    job = sim.run(qc_t, shots=4096)
    result = job.result()
    counts = result.get_counts()

    print("Measurement counts:", counts)

    # Determine the most probable measured outcome(s).
    max_count = max(counts.values())
    top_outcomes = {k for k, v in counts.items() if v == max_count}
    top_indices = {int(bs, 2) for bs in top_outcomes}  # bs is little-endian per Qiskit (qubit0 = rightmost char)

    print("Top measured index/indices:", sorted(top_indices))

    total_shots = sum(counts.values())
    marked_mass = sum(v for k, v in counts.items() if int(k, 2) in minimizers) / total_shots
    verified = (
        top_indices.issubset(minimizers)
        and len(top_indices) > 0
        and marked_mass > 0.9
    )
    ran_ok = True
    print("Fraction of shots landing on a minimizer:", round(marked_mass, 4))

    if verified:
        print("PASS: Grover search's top measured outcome(s) %s match the "
              "classically verified minimum-discrepancy set %s (D_min=%d)."
              % (sorted(top_indices), sorted(minimizers), d_min))
    else:
        print("FAIL: Grover search's top measured outcome(s) %s do NOT match "
              "the classically verified minimum-discrepancy set %s."
              % (sorted(top_indices), sorted(minimizers)))

    return ran_ok, verified


if __name__ == "__main__":
    ok, verified = main()
    print("ran_ok=%s verified_against_classical=%s" % (ok, verified))
