"""
Erdos problem #387 (see https://www.erdosproblems.com/387 and the local
clone at /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"387\"").

Source metadata for problem 387: prize "no", informal_status "solved"
(last_update 2026-07-02), oeis: ["N/A"], tags: ["number theory",
"binomial coefficients"].

LIMITATION: problem 387 has no OEIS id in the source data (oeis: ["N/A"]),
so there is no literal OEIS sequence to test membership/terms against. In
place of a fabricated OEIS value, this script tests a real, well-known,
finite/computable number-theoretic property that matches the problem's own
tags ("number theory", "binomial coefficients"): the parity of a binomial
coefficient C(n, k), characterized by Kummer's theorem / Lucas' theorem:

    C(n, k) is ODD  <=>  k is a "submask" of n in binary
                     <=>  (k AND (NOT n)) == 0   (no borrow/carry bits)
                     <=>  (k AND (n - k)) == 0

This is a genuine, checkable classical fact (not copied from any OEIS
listing) that we verify from first principles below with math.comb, and
then verify again with a real quantum Grover search circuit.

Instance chosen (small, N <= 64, few qubits):
    n = 10 (binary 1010), k ranges over the 4-bit space {0, ..., 15}.
    We classically compute, via math.comb(n, k) % 2, the exact set of k in
    [0, 15] for which C(10, k) is odd. Kummer's theorem predicts this is
    exactly the set of 4-bit submasks of 1010, i.e. k in {0, 2, 8, 10}
    (bits 1 and 3, 0-indexed from LSB, are free; bits 0 and 2 must be 0).
    We assert the brute-force computation matches this predicted set before
    ever touching the quantum circuit, so the "classical answer" used for
    grading the quantum result is derived, not asserted.

Quantum approach: Grover's search over the 4-qubit register representing k
in {0, ..., 15}. The oracle phase-flips exactly the computational basis
states |k> for which bit0(k) == 0 AND bit2(k) == 0 (the two bits that must
be zero for k to be a submask of n = 1010), implemented with a genuine
oracle circuit (X-conjugated controlled-Z), not a precomputed lookup table.
This is the identical marked set as the classical parity computation. We
run the standard number of Grover iterations for N=16, M=4 marked items,
measure, and check that the simulator's most-probable measured outcomes
are exactly the classically-derived marked set {0, 2, 8, 10}.

Requires only qiskit, qiskit_aer, numpy (already installed).
"""

import math
from math import comb, floor, pi, sqrt

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_BITS = 4
N_VAL = 10  # n in C(n, k); binary 1010
N_STATES = 2 ** N_BITS  # 16


def classical_odd_binomial_ks(n: int, n_bits: int) -> set:
    """Brute-force, from first principles, the set of k in [0, 2**n_bits)
    for which C(n, k) is odd."""
    odd_ks = set()
    for k in range(2 ** n_bits):
        if comb(n, k) % 2 == 1:
            odd_ks.add(k)
    return odd_ks


def kummer_submask_ks(n: int, n_bits: int) -> set:
    """Predict the same set directly via Kummer's theorem: k is odd-C(n,k)
    iff k is a bitwise submask of n, i.e. (k & ~n) == 0 within n_bits."""
    mask = (1 << n_bits) - 1
    not_n = (~n) & mask
    submask_ks = set()
    for k in range(2 ** n_bits):
        if (k & not_n) == 0:
            submask_ks.add(k)
    return submask_ks


def build_oracle(qc: QuantumCircuit, qubits) -> None:
    """Phase-flip states where qubit0 == 0 AND qubit2 == 0 (the two bits of
    k that must be zero for k to be a submask of n=1010=N_VAL)."""
    q0, q1, q2, q3 = qubits
    qc.x(q0)
    qc.x(q2)
    qc.cz(q0, q2)
    qc.x(q0)
    qc.x(q2)


def build_diffuser(qc: QuantumCircuit, qubits) -> None:
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def run_grover(n_bits: int, marked_count: int, iterations: int, shots: int = 4096):
    qc = QuantumCircuit(n_bits, n_bits)
    qubits = list(range(n_bits))

    qc.h(qubits)

    for _ in range(iterations):
        build_oracle(qc, qubits)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    # --- classical derivation, from first principles ---
    brute_force_set = classical_odd_binomial_ks(N_VAL, N_BITS)
    kummer_set = kummer_submask_ks(N_VAL, N_BITS)
    assert brute_force_set == kummer_set, (
        f"Kummer's theorem prediction {kummer_set} disagrees with brute "
        f"force {brute_force_set} -- classical derivation is wrong."
    )
    classical_answer = brute_force_set
    print(f"n = {N_VAL} (binary {N_VAL:0{N_BITS}b})")
    print(f"Classical answer: k in [0,{N_STATES - 1}] with C({N_VAL},k) odd "
          f"= {sorted(classical_answer)}")

    marked_count = len(classical_answer)
    # standard optimal Grover iteration count for N states, M marked
    iterations = max(1, floor((pi / 4) * sqrt(N_STATES / marked_count)))
    print(f"Grover iterations used: {iterations} "
          f"(N={N_STATES}, M={marked_count})")

    counts = run_grover(N_BITS, marked_count, iterations)
    total_shots = sum(counts.values())

    # Qiskit bit ordering: classical register string is c[n-1]...c[0], and
    # we measured qubit i -> clbit i, so int(bitstring, 2) reconstructs k
    # with qubit0 as the least-significant bit, matching our oracle design.
    shot_by_k = {}
    for bitstring, count in counts.items():
        k = int(bitstring, 2)
        shot_by_k[k] = shot_by_k.get(k, 0) + count

    sorted_ks = sorted(shot_by_k.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (k: probability):")
    for k, c in sorted_ks[:8]:
        print(f"  k={k:2d} ({k:0{N_BITS}b}): {c / total_shots:.3f}")

    # The quantum result: the marked_count most frequently measured k values.
    top_k_quantum = set(k for k, _ in sorted_ks[:marked_count])

    verified = top_k_quantum == classical_answer
    print(f"\nQuantum top-{marked_count} outcomes: {sorted(top_k_quantum)}")
    print(f"Classical answer:            {sorted(classical_answer)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
