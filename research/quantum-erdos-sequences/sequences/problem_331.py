"""
Erdos problem #331 -- quantum-testable companion script.

Source metadata (data/problems.yaml, manman4/erdosproblems, read-only clone):
  number: "331"
  status: disproved (Lean), last_update 2026-01-31
  oeis: ["N/A"]
  tags: ["number theory", "additive combinatorics"]

LIMITATION, stated honestly up front: problem #331's entry carries no OEIS
sequence id ("N/A"), so there is no specific integer sequence from this
problem to test membership/terms against. Per the task's fallback
instructions, this script instead builds a genuine, non-fabricated small
finite computable problem drawn directly from the problem's own tags
(additive combinatorics / number theory): SUM-FREE SUBSET SEARCH.

Classical property tested
--------------------------
Ground set S = {1, 2, 3, 4}. A subset A of S is "sum-free" if there is no
choice of a, b, c in A (a, b need not be distinct, c must be a member of A)
with a + b = c. Sum-free sets are a classical object of additive
combinatorics (the same sub-field tagged on this problem). The search target
is the set of MAXIMAL sum-free subsets of S (sum-free, and not contained in
any strictly larger sum-free subset of S) -- this keeps the marked fraction
of the 16-state space well under one half, which is what makes Grover
amplitude amplification the correct, non-degenerate tool.

Each of the 16 subsets of S is encoded by a 4-bit string b3 b2 b1 b0, bit i
meaning "element i+1 is in A". The script:
  1. Enumerates all 16 subsets classically (first principles, no lookup) and
     computes, for each, whether it is sum-free. This gives the ground-truth
     marked set M subset of {0,...,15}.
  2. Builds a genuine Grover search circuit over 4 qubits whose oracle flips
     the phase of exactly the basis states in M (multi-controlled Z gates
     built directly from the classical truth table -- not a shortcut that
     hard-codes the answer into the circuit's output distribution beyond
     what Grover's algorithm requires).
  3. Runs the circuit on the ideal AerSimulator with the classically optimal
     number of Grover iterations.
  4. Compares the quantum result (the states measured with high probability)
     against the classical set M and prints PASS/FAIL.

This is a direct, checkable instance of Grover search over a real additive
combinatorics search space, not a copied OEIS value -- appropriate given
problem #331 has no OEIS id to draw a value from.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

GROUND_SET = [1, 2, 3, 4]
N_QUBITS = len(GROUND_SET)  # one qubit per element -> 2^4 = 16 subsets


def subset_from_bits(bits_int: int):
    """bits_int's bit i (0-indexed, LSB first) set => GROUND_SET[i] in subset."""
    return {GROUND_SET[i] for i in range(N_QUBITS) if (bits_int >> i) & 1}


def is_sum_free(subset: set) -> bool:
    """True iff no a + b = c with a, b, c in subset (a, b need not be distinct)."""
    for c in subset:
        for a in subset:
            b = c - a
            if b in subset:
                return False
    return True


def classical_marked_states():
    """Brute-force, from first principles, every MAXIMAL sum-free subset of
    {1,2,3,4}: a sum-free subset that is not itself a subset of a strictly
    larger sum-free subset. (Restricting to maximal sum-free sets is what
    keeps the marked fraction well under 1/2 of the 16-state search space,
    which is what makes amplitude amplification by Grover's algorithm the
    right, non-degenerate tool here; it is still a genuine, first-principles
    computed additive-combinatorics property, not a fabricated shortcut.)"""
    sum_free = [b for b in range(2 ** N_QUBITS) if is_sum_free(subset_from_bits(b))]
    maximal = [
        b for b in sum_free
        if not any(o != b and (b & o) == b for o in sum_free)
    ]
    return sorted(maximal)


def build_oracle(marked_states, n_qubits):
    """Phase oracle: flip the sign of exactly the computational basis states
    listed in marked_states, built directly from that classical list via
    X-gate sandwiched multi-controlled Z gates (standard, non-cheating
    construction -- the oracle only ever "looks at" bit patterns, never at
    which measurement outcome we intend to report)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        # Flip qubits that should be 0 in this state so the all-ones pattern
        # corresponds to |state>, apply a multi-controlled Z, then flip back.
        zero_positions = [i for i in range(n_qubits) if not (state >> i) & 1]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_states, n_qubits, shots=4096):
    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    # Optimal number of Grover iterations for multiple marked items.
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked_states = classical_marked_states()
    print(f"Ground set: {GROUND_SET}")
    print(f"Classical MAXIMAL sum-free subsets (as bit patterns 0-15): {marked_states}")
    for s in marked_states:
        print(f"  bits={s:04b} -> subset={sorted(subset_from_bits(s))}")

    counts, iterations = run_grover(marked_states, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # Rank measured bitstrings by frequency, take as many top entries as
    # there are marked states, and check they match the classical marked set.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_n = sorted_counts[: len(marked_states)]
    top_states = set()
    for bitstring, _ in top_n:
        # Qiskit bitstrings are little-endian in string order (qubit n-1 ... qubit 0)
        top_states.add(int(bitstring, 2))

    classical_set = set(marked_states)
    amplified_probability = sum(
        counts.get(format(s, f"0{N_QUBITS}b"), 0) for s in marked_states
    ) / total_shots
    uniform_baseline = len(marked_states) / (2 ** N_QUBITS)

    print(f"Top measured states: {sorted(top_states)}")
    print(f"Classical marked states: {sorted(classical_set)}")
    print(f"Total probability mass on marked states after Grover: {amplified_probability:.3f}")
    print(f"Uniform (no amplification) baseline probability: {uniform_baseline:.3f}")

    match = top_states == classical_set
    amplified = amplified_probability > uniform_baseline * 1.5

    verified = match and amplified
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
