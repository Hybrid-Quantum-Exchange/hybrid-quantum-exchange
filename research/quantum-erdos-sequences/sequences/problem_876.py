"""
Erdos problem #876 -- quantum-testable sequence lane.

LIMITATION (read first): problem #876 (additive combinatorics, informal
status "open", https://www.erdosproblems.com/876) carries no OEIS id in the
source data file (data/problems.yaml lists `oeis: ["N/A"]`). There is
therefore no concrete integer sequence attached to this problem number to
derive a property from. This script is an honest best-effort substitute,
not a faithful encoding of problem #876 itself: it builds a real, genuine
quantum circuit (Grover search) for a small, finite, computable property
from the same subject area the problem is tagged with -- additive
combinatorics -- namely: existence of a Sidon set (a "B2 set", a set with
all pairwise sums distinct) of size 3 inside {0, 1, ..., 7}.

Classical property tested
--------------------------
Search space: all 4-element subsets {a, b, c, d} of Z_6 = {0,...,5},
encoded as four 3-qubit registers (a, b, c, d), with a < b < c < d
enforced by the oracle (only ordered/sorted patterns are ever marked).
A quadruple is a "hit" iff it is a genuine Sidon set (B2 set): all
pairwise sums a+b, a+c, a+d, b+c, b+d, c+d are distinct. Unlike a
3-element ordered subset (where a<b<c forces a+b<a+c<b+c automatically,
making "Sidon" trivially true for every triple), 4-element subsets have
a real, non-trivial failure mode -- the classic collision a+d == b+c --
so this is genuine combinatorial content, not a vacuous predicate. This
is exactly the finite object at the heart of Sidon-set / additive
combinatorics questions, the same subject area Erdos problem #876 is
tagged with, restricted to a tiny instance (C(6,4) = 15 candidates) that
a 12-qubit circuit can search exhaustively.

The script first computes, by brute-force classical enumeration in
first-principles Python (no external data, no OEIS lookup), the exact
list of Sidon quadruples in {0,...,5} and their count (8 of the 15
4-subsets, verified below). It then builds a Grover search circuit whose
oracle marks exactly those quadruples (a classical reversible oracle
compiled directly from the truth table of the same classical predicate),
runs it on the ideal AerSimulator, and checks that Grover amplification
concentrated probability on genuine Sidon quadruples.

PASS criterion: after running the Grover circuit, at least one of the
states with amplified measurement count (count > 2x the uniform-random
expectation) is a classically-verified Sidon quadruple, and every
amplified outcome that appears among the top measured outcomes is a
true classical solution (no false positives leak into the amplified set).
"""

import itertools
import math

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N = 6          # elements 0..5 -> 3 qubits each register (values 0-5 fit in 3 bits)
NBITS = 3
K = 4          # subset size


def classical_sidon_quadruples(n=N):
    """Brute-force, first-principles: all {a<b<c<d} in {0..n-1} that are Sidon
    (B2) sets, i.e. all 6 pairwise sums are distinct."""
    hits = []
    for a, b, c, d in itertools.combinations(range(n), K):
        sums = {a + b, a + c, a + d, b + c, b + d, c + d}
        if len(sums) == 6:
            hits.append((a, b, c, d))
    return hits


def build_oracle(qc, regs, hits):
    """
    Phase oracle that flips the sign of exactly the basis states in `hits`.

    For each classical hit (a,b,c,d), wrap the "0" bits with X gates so the
    target pattern becomes all-ones, apply a phase-kickback multi-controlled
    Z across all qubits, then undo the X gates. This is a direct,
    brute-force but fully correct classical-to-quantum oracle compiled from
    the truth table (valid here because the search space is tiny: 6^4 =
    1296 basis states over 12 qubits, and only 15 of them are even
    sorted/valid quadruples).
    """
    all_qubits = [q for reg in regs for q in reg]
    for values in hits:
        bits = []
        for val in values:
            for i in range(NBITS):
                bits.append((val >> i) & 1)
        flip_qubits = [q for q, bit in zip(all_qubits, bits) if bit == 0]
        if flip_qubits:
            qc.x(flip_qubits)
        target = all_qubits[-1]
        controls = all_qubits[:-1]
        qc.h(target)
        qc.append(MCXGate(len(controls)), controls + [target])
        qc.h(target)
        if flip_qubits:
            qc.x(flip_qubits)


def build_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    target = qubits[-1]
    controls = qubits[:-1]
    qc.h(target)
    qc.append(MCXGate(len(controls)), controls + [target])
    qc.h(target)
    qc.x(qubits)
    qc.h(qubits)


def run_grover(hits, n=N, shots=8192):
    regs = [QuantumRegister(NBITS, name) for name in ("a", "b", "c", "d")]
    creg = ClassicalRegister(K * NBITS, "m")
    qc = QuantumCircuit(*regs, creg)

    all_qubits = [q for reg in regs for q in reg]
    qc.h(all_qubits)

    N_states = n ** K
    M = len(hits)
    if M == 0:
        raise RuntimeError("no classical solutions found -- cannot demonstrate amplification")

    theta = math.asin(math.sqrt(M / N_states))
    iterations = max(1, round((math.pi / 4) / theta - 0.5))

    for _ in range(iterations):
        build_oracle(qc, regs, hits)
        build_diffuser(qc, all_qubits)

    qc.measure(all_qubits, creg)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def decode_bitstring(bitstring):
    """Qiskit prints classical-register bits with bit 0 rightmost, and the
    creg was measured in qubit order [a0 a1 a2 b0 b1 b2 c0 c1 c2 d0 d1 d2]."""
    bits = bitstring.replace(" ", "")[::-1]
    parts = [bits[i * NBITS:(i + 1) * NBITS] for i in range(K)]

    def to_int(s):
        v = 0
        for i, ch in enumerate(s):
            v |= int(ch) << i
        return v

    return tuple(to_int(p) for p in parts)


def main():
    hits = classical_sidon_quadruples(N)
    hit_set = set(hits)
    total_subsets = math.comb(N, K)
    print(f"Classical brute force: {len(hits)} Sidon (B2) quadruples among "
          f"{total_subsets} 4-subsets of Z_{N}.")
    print("All Sidon quadruples:", hits)

    counts, iterations = run_grover(hits, N)
    print(f"Grover iterations used: {iterations}")

    decoded_counts = {}
    for bitstring, cnt in counts.items():
        quad = decode_bitstring(bitstring)
        decoded_counts[quad] = decoded_counts.get(quad, 0) + cnt

    total_shots = sum(decoded_counts.values())
    uniform_expected = total_shots / (N ** K)

    ranked = sorted(decoded_counts.items(), key=lambda kv: -kv[1])
    top_k = ranked[: max(1, min(len(hits), 10))]

    amplified_and_valid = 0
    amplified_total = 0
    for quad, cnt in ranked:
        if cnt > uniform_expected * 2:
            amplified_total += 1
            if quad in hit_set:
                amplified_and_valid += 1

    print(f"Top measured outcomes: {top_k}")
    print(f"Amplified outcomes (count > 2x uniform): {amplified_total}, "
          f"of which valid Sidon quadruples: {amplified_and_valid}")

    found_valid_in_top = any(t in hit_set for t, _ in top_k)
    no_false_positives_amplified = amplified_and_valid == amplified_total

    passed = (
        found_valid_in_top
        and amplified_total > 0
        and amplified_and_valid >= 1
        and no_false_positives_amplified
    )

    if passed:
        print("PASS")
    else:
        print("FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
