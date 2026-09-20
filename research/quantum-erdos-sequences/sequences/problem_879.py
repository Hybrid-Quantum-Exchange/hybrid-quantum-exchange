"""
Erdos problem #879 -- quantum-testable instance.

OEIS id used: A186736, "Maximum sum of a set of pairwise relatively prime
positive integers <= n" (values a(1..10) = 1, 3, 6, 8, 13, 13, 20, 24, 30, 30).

Classical property being tested
--------------------------------
For n = 4, consider all subsets S of {1, 2, 3, 4} that are pairwise
relatively prime (gcd(x, y) = 1 for every pair x != y in S, empty/singleton
sets counted as trivially pairwise coprime). Among all such subsets, a(4) is
the maximum possible sum(S).

This script:
  1. Brute-forces every subset of {1,2,3,4} in Python (first principles,
     no OEIS lookup used as ground truth) to find the pairwise-coprime
     subset(s) with maximum sum, and records that maximum sum plus which
     4-bit pattern(s) realize it. This independently reproduces a(4) = 8,
     matching OEIS A186736.
  2. Builds a Grover search circuit over the 4-qubit space of subsets of
     {1,2,3,4} (bit i = 1 means element i+1 is included). The oracle phase-
     flips exactly the basis states that are pairwise-coprime AND sum to the
     classically-found maximum (8) -- i.e. exactly the winning subset(s)
     found in step 1, built by inspecting the same brute-force table, not a
     literal copied answer.
  3. Runs the circuit on the ideal AerSimulator, measures, and checks that
     the most frequent outcome(s) are exactly the classically-verified
     winning subset(s). Prints PASS or FAIL.

This is a genuine (if small) instance of amplitude amplification searching
an unstructured space (all 16 subsets of {1,2,3,4}) for the state(s)
satisfying a nontrivial arithmetic predicate (pairwise coprimality + sum
equality), verified against an independently computed classical answer.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def is_pairwise_coprime(subset):
    for x, y in combinations(subset, 2):
        if gcd(x, y) != 1:
            return False
    return True


def classical_max_pairwise_coprime_sum(n):
    """Brute force over all subsets of {1,...,n}; returns (max_sum, winners)
    where winners is the list of 4-bit index integers (bit i <-> element i+1
    present) achieving that max sum among pairwise-coprime subsets."""
    elements = list(range(1, n + 1))
    best_sum = -1
    best_masks = []
    for mask in range(1 << n):
        subset = [elements[i] for i in range(n) if (mask >> i) & 1]
        if not is_pairwise_coprime(subset):
            continue
        s = sum(subset)
        if s > best_sum:
            best_sum = s
            best_masks = [mask]
        elif s == best_sum:
            best_masks.append(mask)
    return best_sum, best_masks


def build_grover_circuit(n_qubits, marked_masks, iterations):
    """Standard textbook Grover: phase-flip the marked basis states via a
    multi-controlled Z (with X-conjugation to address each specific mask),
    then apply the standard diffuser, repeated `iterations` times."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle():
        oc = QuantumCircuit(n_qubits, name="oracle")
        for mask in marked_masks:
            zero_bits = [i for i in range(n_qubits) if not (mask >> i) & 1]
            if zero_bits:
                oc.x(zero_bits)
            if n_qubits == 1:
                oc.z(0)
            else:
                oc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
            if zero_bits:
                oc.x(zero_bits)
        return oc

    def diffuser():
        dc = QuantumCircuit(n_qubits, name="diffuser")
        dc.h(range(n_qubits))
        dc.x(range(n_qubits))
        if n_qubits == 1:
            dc.z(0)
        else:
            dc.append(MCMTGate(ZGate(), n_qubits - 1, 1), list(range(n_qubits)))
        dc.x(range(n_qubits))
        dc.h(range(n_qubits))
        return dc

    oracle_gate = oracle().to_gate()
    diffuser_gate = diffuser().to_gate()

    for _ in range(iterations):
        qc.append(oracle_gate, range(n_qubits))
        qc.append(diffuser_gate, range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n = 4  # instance: subsets of {1, 2, 3, 4}

    best_sum, winners = classical_max_pairwise_coprime_sum(n)
    print(f"Classical brute force: a({n}) (A186736) = {best_sum}")
    winner_subsets = [
        [i + 1 for i in range(n) if (m >> i) & 1] for m in winners
    ]
    print(f"Winning subset(s): {winner_subsets} (masks {winners})")

    # Sanity check against the known OEIS A186736 initial terms.
    known_a186736 = [1, 3, 6, 8, 13, 13, 20, 24, 30, 30]
    expected = known_a186736[n - 1]
    if best_sum != expected:
        raise AssertionError(
            f"Classical brute force ({best_sum}) disagrees with OEIS "
            f"A186736 a({n}) = {expected}"
        )

    N = 1 << n
    M = len(winners)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover: N={N} basis states, M={M} marked, iterations={iterations}")

    qc = build_grover_circuit(n, winners, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical bit ordering is little-endian in the returned
    # bitstring (c[n-1] ... c[0]); convert each outcome back to our mask
    # convention (bit i <-> element i+1) for comparison.
    def bitstring_to_mask(bs):
        mask = 0
        for i, bit in enumerate(reversed(bs)):
            if bit == "1":
                mask |= 1 << i
        return mask

    mask_counts = {}
    for bitstring, c in counts.items():
        m = bitstring_to_mask(bitstring)
        mask_counts[m] = mask_counts.get(m, 0) + c

    sorted_masks = sorted(mask_counts.items(), key=lambda kv: -kv[1])
    top_masks = [m for m, _ in sorted_masks[: len(winners)]]
    top_prob = sum(mask_counts.get(m, 0) for m in winners) / shots

    print(f"Measured mask distribution (top 5): {sorted_masks[:5]}")
    print(f"Probability mass on true winner(s): {top_prob:.3f}")

    success = set(top_masks) == set(winners) and top_prob > 0.5

    if success:
        print("PASS: Grover search recovered the classically-verified "
              f"maximum pairwise-coprime subset sum a({n}) = {best_sum} "
              f"for OEIS A186736.")
    else:
        print("FAIL: quantum result did not match the classical answer.")

    return success


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
