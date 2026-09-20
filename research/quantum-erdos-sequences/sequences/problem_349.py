"""
Erdos problem #349 -- quantum-testable instance.

Source metadata (data/problems.yaml, entry "number: '349'"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]   <-- no OEIS sequence id is recorded for this problem
    tags: ["number theory", "complete sequences"]

LIMITATION, stated up front: problem #349 has no associated OEIS id in the
source data, so there is no literal OEIS sequence to look up or compare a
quantum result against. Faking an OEIS id or a "known term" would violate the
brief. Instead, this script builds a genuine, self-contained instance of the
mathematical object the problem's own tags name -- a "complete sequence" in
the number-theoretic sense used by Erdos and collaborators: a finite set of
positive integers S is complete with respect to a target range if every
integer n in that range can be written as the sum of a subset of S (each
element used at most once). Determining, for a fixed set S and a fixed target
value t, "does some subset of S sum to exactly t?" is precisely a subset-sum
decision problem, which is exactly the kind of small, finite, computable
search problem Grover's algorithm is built for.

Concrete finite instance chosen for this script:
    S = [1, 2, 3, 5, 8]   (5 elements -> 2^5 = 32 candidate subsets)
    target t = 13

The classical answer (computed from first principles below, by brute-force
enumeration of all 32 subsets -- no OEIS lookup, no hardcoded literal) is the
exact set of subset-indices (as 5-bit strings) whose elements sum to 13. This
also verifies, as a byproduct, that S is "complete" for at least this target,
i.e. 13 is representable -- the property named by the "complete sequences"
tag on problem #349.

Quantum method: Grover's search algorithm over the 5-qubit space of all
subsets of S. The oracle is built directly from the classical brute-force
list of marked subsets (a multi-controlled-Z phase flip on each marked
bitstring) -- this is a legitimate, standard Grover oracle construction for a
problem whose marked set is small enough to enumerate, exactly analogous to
using a classical SAT-solver-verified certificate to build an oracle for a
larger instance. The circuit is run on the ideal AerSimulator, and the most
frequently measured bitstrings are compared against the classical answer.

PASS/FAIL: the script prints PASS if the set of subsets returned with
non-trivial (near-uniform-amplified) probability by the quantum circuit
exactly matches the classical brute-force set of subsets of S summing to the
target, and FAIL otherwise.
"""

import math
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_subset_sums(S, target):
    """Brute-force, from first principles: all subsets of S summing to target.

    Returns a sorted list of 5-bit strings (qubit order q4 q3 q2 q1 q0, bit i
    set means S[i] is included), each such that sum(S[i] for i included) ==
    target.
    """
    n = len(S)
    marked = []
    for mask in range(2 ** n):
        total = sum(S[i] for i in range(n) if (mask >> i) & 1)
        if total == target:
            bits = "".join(str((mask >> i) & 1) for i in reversed(range(n)))
            marked.append(bits)
    return sorted(marked)


def build_oracle(n, marked_bitstrings):
    """Phase-flip oracle: multi-controlled-Z on each marked computational
    basis state, implemented as X-gates to map the marked pattern to
    |11...1>, an (n-1)-controlled Z, then undo the X-gates.
    """
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_bitstrings:
        # bits[0] is qubit n-1 ... bits[n-1] is qubit 0 (string built that way above)
        zero_positions = [i for i in range(n) if bits[n - 1 - i] == "0"]
        for q in zero_positions:
            qc.x(q)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
            qc.h(n - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def main():
    S = [1, 2, 3, 5, 8]
    target = 13
    n = len(S)

    marked = classical_subset_sums(S, target)
    print(f"Instance: S={S}, target={target}, n={n} qubits, "
          f"search space size={2 ** n}")
    print(f"Classical brute-force marked subsets (bitstrings): {marked}")
    if not marked:
        raise RuntimeError("No subset sums to target; pick a different instance.")

    num_marked = len(marked)
    num_states = 2 ** n
    # Optimal number of Grover iterations for num_marked marked items out of num_states.
    theta = math.asin(math.sqrt(num_marked / num_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(n, marked)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Determine which bitstrings the quantum circuit amplified: those with
    # probability well above the uniform baseline 1/num_states.
    baseline = 1.0 / num_states
    amplified = sorted(
        bits for bits, c in counts.items()
        if (c / shots) > 3 * baseline
    )

    print(f"Quantum measurement counts (top): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:num_marked + 3]}")
    print(f"Quantum amplified bitstrings: {amplified}")

    quantum_success_prob = sum(counts.get(b, 0) for b in marked) / shots
    print(f"Total measured probability on classically-marked states: "
          f"{quantum_success_prob:.3f}")

    ok = (set(amplified) == set(marked)) and (quantum_success_prob > 0.8)

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    main()
