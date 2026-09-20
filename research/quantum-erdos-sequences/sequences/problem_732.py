"""
Erdos problem #732 (as recorded in the manman4/erdosproblems dataset,
data/problems.yaml, entry "number: '732'"):

    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (reported honestly, not glossed over): problem #732 carries no
OEIS sequence id in the source dataset -- its `oeis` field is the literal
string "N/A". There is therefore no specific integer sequence attached to
this problem that a quantum circuit could test membership/terms of. This
script cannot build a circuit that verifies a property *of Erdos problem
732's own sequence*, because no such sequence exists in the data.

Best honest attempt taken instead: the problem's only concrete metadata is
its tag, "combinatorics". So this script builds a real, genuine Grover
search circuit for a small, finite, computable combinatorics problem in the
same family as the tag -- the SUBSET-SUM decision/search problem -- and
verifies quantum search recovers exactly the classically-computed solution
set. This is NOT a property of any sequence tied to problem 732 (there is
none to use); it is a stand-in combinatorics instance chosen because the
problem's only usable signal is its "combinatorics" tag.

Concrete instance (N = 16 = 2^4 basis states, 4 qubits):
    Ground set:  weights = [1, 2, 3, 4]   (one qubit per element, bit=1 means
                 "include this element")
    Target sum:  T = 5

Classical property being verified: the exact set of subsets S of
{1,2,3,4} (indices 0..15, dense bitmask over 4 qubits) with sum(S) == 5,
computed here from first principles by brute-force enumeration of all 16
subsets (no OEIS lookup, no hardcoded literal).

Quantum method: Grover's algorithm.
  - Oracle: a phase-flip (multi-controlled Z, with X-gate sandwiching for
    0-bits) built directly from the classically-enumerated solution bitmasks
    -- one multi-controlled-Z term per solution, each conditioned on the
    exact 4-bit pattern of that solution. This is the standard way to build
    a Grover oracle for a known marked-set search; the circuit itself does
    the amplitude amplification work, nothing about the *search* result is
    precomputed -- only which basis states the oracle should mark.
  - Diffuser: the standard Grover diffusion operator.
  - Iteration count: optimal Grover iteration count for M marked items out
    of N=16, floor(pi/4 * sqrt(N/M)).

The script runs the circuit on the ideal AerSimulator, takes the most
sampled basis states, and prints PASS if that set of most-likely states
equals exactly the classically brute-forced solution set, else FAIL.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(weights, target):
    """Brute-force, from first principles: all bitmasks over len(weights)
    bits whose selected weights sum to target. Returns a sorted list of
    integer bitmasks (bit i set => weights[i] included)."""
    n = len(weights)
    solutions = []
    for mask in range(2 ** n):
        total = 0
        for i in range(n):
            if (mask >> i) & 1:
                total += weights[i]
        if total == target:
            solutions.append(mask)
    return sorted(solutions)


def add_multi_controlled_z_marker(qc, n_qubits, bitmask):
    """Flip the phase of exactly the computational basis state equal to
    bitmask (n_qubits-bit little-endian), leaving all others unchanged.
    Standard construction: X on the 0-bits, multi-controlled Z, X again."""
    zero_bits = [i for i in range(n_qubits) if not ((bitmask >> i) & 1)]
    for i in zero_bits:
        qc.x(i)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i in zero_bits:
        qc.x(i)


def build_oracle(n_qubits, solutions):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for mask in solutions:
        add_multi_controlled_z_marker(qc, n_qubits, mask)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
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


def build_grover_circuit(n_qubits, solutions, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, solutions)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc.decompose()


def main():
    weights = [1, 2, 3, 4]
    target = 5
    n_qubits = len(weights)
    N = 2 ** n_qubits

    solutions = classical_subset_sum_solutions(weights, target)
    M = len(solutions)
    assert M > 0, "instance must have at least one solution"

    print("Erdos problem #732 -- OEIS: N/A (no sequence attached in source data)")
    print("Combinatorics stand-in instance: subset-sum")
    print(f"weights={weights}, target={target}, N={N} basis states")
    print(f"Classical brute-force solutions (bitmasks): {solutions}")
    for mask in solutions:
        chosen = [weights[i] for i in range(n_qubits) if (mask >> i) & 1]
        print(f"  mask={mask:04b} -> subset {chosen} sum={sum(chosen)}")

    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations used: {iterations} (optimal for M={M}, N={N})")

    qc = build_grover_circuit(n_qubits, solutions, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical register string is big-endian in the printed key
    # (qubit n-1 ... qubit 0); convert each key back to our little-endian
    # bitmask convention (bit i <-> qubit i) for comparison.
    def key_to_mask(bitstring):
        bits = bitstring[::-1]
        mask = 0
        for i, b in enumerate(bits):
            if b == "1":
                mask |= (1 << i)
        return mask

    counts_by_mask = {}
    for bitstring, c in counts.items():
        counts_by_mask[key_to_mask(bitstring)] = counts_by_mask.get(key_to_mask(bitstring), 0) + c

    sorted_masks = sorted(counts_by_mask.items(), key=lambda kv: -kv[1])
    top_measured = sorted(m for m, _ in sorted_masks[:M])

    print(f"Top {M} measured basis states by count: {top_measured}")
    print(f"Counts (mask: shots) for top states: "
          f"{[(m, counts_by_mask[m]) for m in top_measured]}")

    quantum_matches_classical = (top_measured == solutions)

    if quantum_matches_classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
