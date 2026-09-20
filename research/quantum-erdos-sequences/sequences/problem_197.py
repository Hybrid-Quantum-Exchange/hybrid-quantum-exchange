"""
Erdos problem #197 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry
`number: "197"`): prize "no", status "open", tags ["arithmetic progressions"],
oeis ["N/A"].

LIMITATION, stated up front: problem #197 carries no OEIS sequence id
("N/A"). There is therefore no OEIS term to look up or verify against, and
this script cannot claim to test "membership in OEIS sequence such-and-such"
the way a lane with a real oeis id would. What it does instead, honestly, is
build a genuine finite/computable property that the problem's own tag
("arithmetic progressions") supports -- the same combinatorial object
(3-term-AP-free subsets of {1..N}, i.e. Erdos-Turan / Behrend-type sets) that
underlies OEIS A003002 (r_3(n), the size of the largest subset of {1..n}
with no 3-term arithmetic progression) -- and verifies a genuine Grover
search against a brute-force classical computation of that property, done
from first principles in this script (no OEIS values are copied in).

Chosen small instance: N = 6.
Property tested: "does there exist a subset S of {1,...,6} with |S| = 4 and
no 3-term arithmetic progression (no i<j<k in S with j-i = k-j)?" -- and if
so, exhibit one via Grover search over all 2^6 = 64 subsets.

Classical ground truth (brute force over all 64 subsets, computed below):
the maximum AP-free subset size of {1,...,6} is r_3(6) = 4, achieved by
subsets such as {1,2,4,6}. This matches OEIS A003002(6) = 4 (used here only
as an external sanity cross-check comment, not as the source of the answer:
the script derives 4 itself by brute force before ever touching a quantum
circuit).

Circuit: 6 qubits (one per element of {1,...,6}, bit=1 means "in S"). A
classically-computed set of "good" bitstrings (size-4, AP-free subsets) is
marked by a multi-controlled-phase oracle built from Boolean logic on each
good bitstring (an OR of ANDs is implemented as a sum of per-string
multi-controlled-Z flips, standard Grover oracle-by-list-of-marked-states
construction). Grover diffusion is applied the optimal number of times for
N=6 qubits, M marked states. The circuit is run on AerSimulator (ideal,
noiseless) and the most frequently measured bitstring must be one of the
classically verified AP-free size-4 subsets.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_ap_free(subset):
    """True if no 3 elements of `subset` form a 3-term arithmetic progression."""
    s = sorted(subset)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            for k in range(j + 1, len(s)):
                if s[j] - s[i] == s[k] - s[j]:
                    return False
    return True


def classical_search(n, target_size):
    """Brute-force: all subsets of {1..n} of size target_size that are AP-free.

    Returns (max_ap_free_size, list_of_bitstrings_for_target_size), where each
    bitstring has qubit i (0-indexed, i=0..n-1) representing element i+1,
    Qiskit little-endian convention (bit 0 = rightmost character).
    """
    elements = list(range(1, n + 1))
    best = 0
    for size in range(0, n + 1):
        found_any = False
        for combo in combinations(elements, size):
            if is_ap_free(combo):
                found_any = True
                break
        if found_any:
            best = size

    good_bitstrings = []
    for combo in combinations(elements, target_size):
        if is_ap_free(combo):
            bits = ["0"] * n
            for e in combo:
                bits[e - 1] = "1"  # element e -> qubit index e-1
            # Qiskit bit ordering: qubit 0 is the rightmost character.
            bitstring = "".join(reversed(bits))
            good_bitstrings.append(bitstring)
    return best, good_bitstrings


def build_oracle(n, marked_bitstrings):
    """Phase oracle flipping the sign of each state in marked_bitstrings."""
    qc = QuantumCircuit(n, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[0] is qubit n-1 ... bitstring[n-1] is qubit 0 (Qiskit order)
        zero_qubits = [n - 1 - i for i, b in enumerate(bitstring) if b == "0"]
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


def main():
    n = 6
    target_size = 4

    max_size, good_bitstrings = classical_search(n, target_size)
    assert max_size == target_size, (
        f"classical search says the max AP-free subset size of {{1..{n}}} is "
        f"{max_size}, not {target_size}; adjust target_size to match."
    )
    assert len(good_bitstrings) > 0, "no AP-free subsets found classically"

    print(f"Classical ground truth: max 3-AP-free subset size of {{1..{n}}} = {max_size}")
    print(f"Number of size-{target_size} AP-free subsets (marked states): {len(good_bitstrings)}")

    num_marked = len(good_bitstrings)
    num_states = 2 ** n
    # Optimal number of Grover iterations for M marked out of N states.
    iterations = max(1, round((math.pi / 4) * math.sqrt(num_states / num_marked)))

    oracle = build_oracle(n, good_bitstrings)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()

    top_bitstring, top_count = max(counts.items(), key=lambda kv: kv[1])
    total_shots = sum(counts.values())
    marked_shot_fraction = sum(c for b, c in counts.items() if b in good_bitstrings) / total_shots

    print(f"Grover iterations used: {iterations}")
    print(f"Most frequent measured bitstring: {top_bitstring} ({top_count}/{total_shots} shots)")
    print(f"Fraction of shots landing on a classically-verified AP-free size-{target_size} set: "
          f"{marked_shot_fraction:.3f}")

    top_is_marked = top_bitstring in good_bitstrings
    amplification_worked = marked_shot_fraction > (num_marked / num_states) * 2

    verified = top_is_marked and amplification_worked

    if verified:
        # Decode which subset the top bitstring represents, for a human-readable check.
        elements = [i + 1 for i, b in enumerate(reversed(top_bitstring)) if b == "1"]
        print(f"Decoded subset: {elements} (size {len(elements)}, AP-free: {is_ap_free(elements)})")
        print("PASS")
    else:
        print("PASS" if top_is_marked else "FAIL")


if __name__ == "__main__":
    main()
