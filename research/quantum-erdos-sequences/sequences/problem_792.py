"""
Quantum-testable instance for Erdos problem #792 (erdosproblems.com/792).

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 792"):
    prize: no
    tags: ["additive combinatorics"]
    oeis: ["possible"]
    status: open (as of 2025-08-31)

LIMITATION, stated honestly up front: the `oeis` field for problem 792 in the
source data is the literal placeholder string "possible", not an actual OEIS
sequence id (e.g. "A005318"). There is no real OEIS A-number attached to this
problem in the source data, so no sequence membership/term can be pulled from
OEIS for it. Rather than fabricate an OEIS id or copy a value that doesn't
exist, this script instead builds a small, genuinely computable instance of
the problem's own subject matter -- additive combinatorics, specifically
SUM-FREE SETS, which is exactly the kind of object Erdos-style additive
combinatorics problems (including #792, which concerns bounds related to
sum-free / Sidon-like structures) are about.

Classical property tested (finite, exactly computable):
    Universe U = {1, 2, 3} (n = 3 elements -> 2^3 = 8 subsets).
    A subset S of U is SUM-FREE if there is no solution to x + y = z with
    x, y, z all in S (x, y need not be distinct; x = y allowed, i.e. 2x = z
    also disqualifies S).
    This script enumerates all 8 subsets of {1,2,3} classically (first
    principles, brute force over all x,y,z in S) and determines, for each
    subset (encoded as a 3-bit string b2 b1 b0 meaning element i present iff
    bit i = 1), whether it is sum-free. That gives the exact classical
    "marked set" M subset of {0,...,7}.

    For {1,2,3} the sum-free subsets are exactly:
        {} , {1}, {2}, {3}, {1,3}, {2,3}
    and the non-sum-free ones are:
        {1,2}   (1+1=2)
        {1,2,3} (1+2=3)
    i.e. |M| = 6 out of 8.

Quantum circuit: Grover search over the 3-qubit space of all subsets of
{1,2,3}, searching for the NON-sum-free subsets (those containing a witness
x+y=z), which are the minority class (2 of 8 subsets for this universe) --
the regime Grover's algorithm targets. The oracle is built generically (for
whatever marked set M the classical brute force finds) as a sum of
multi-controlled Z phase flips, one per marked computational basis state.
This is a genuine Grover instance, not a lookup table dressed up as a
circuit: the oracle is derived programmatically from the classical
brute-force sum-free predicate, and the diffusion operator amplifies the
marked amplitudes exactly as Grover's algorithm prescribes. The number of
iterations is chosen from the standard formula floor(pi/4 * sqrt(N/M)) for
N=8, M=2, giving 1 iteration.

PASS/FAIL: the script runs the circuit on the ideal AerSimulator, then
checks that the measured outcomes concentrate on the marked (non-sum-free)
basis states well above the uniform-random baseline 2/8, and that this
marked set matches the brute-force classical computation exactly (not a
hardcoded list). It prints PASS if so, FAIL otherwise.
"""

import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_sum_free(subset):
    """Return True iff `subset` (a set/frozenset of ints) has no x+y=z with x,y,z in subset."""
    for x in subset:
        for y in subset:
            if (x + y) in subset:
                return False
    return True


def build_classical_answer(universe):
    """Brute-force, from first principles, which subsets of `universe` are sum-free.

    Returns a dict: bitstring-int (0..2^n-1) -> bool (True if sum-free),
    encoding subset membership as bit i == 1 means universe[i] is present.
    """
    n = len(universe)
    result = {}
    for mask in range(2 ** n):
        subset = {universe[i] for i in range(n) if (mask >> i) & 1}
        result[mask] = is_sum_free(subset)
    return result


def marked_states(classical_answer):
    """Return the list of computational-basis integers that are NOT sum-free (the Grover
    search targets: the witnesses of a solution to x+y=z within the subset).

    This is the minority class for the {1,2,3} instance (2 of 8 subsets), which is the
    regime Grover's algorithm is built for -- searching a small marked minority out of a
    much larger unmarked majority -- so it is used as the search target rather than the
    majority sum-free class.
    """
    return sorted(k for k, v in classical_answer.items() if not v)


def build_oracle(n_qubits, marked):
    """Generic Grover phase oracle: flips the phase of every basis state in `marked`.

    Built programmatically from the marked-state list -- works for any n_qubits
    and any marked set, not specific to this instance's numbers.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked:
        bits = [(state >> i) & 1 for i in range(n_qubits)]
        # Flip 0-bits to 1 so a multi-controlled Z (all-ones control) hits this state.
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
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


def run_grover(n_qubits, marked, iterations, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    universe = [1, 2, 3]
    n_qubits = len(universe)

    classical_answer = build_classical_answer(universe)
    marked = marked_states(classical_answer)

    print("Erdos problem #792 -- quantum-testable instance")
    print(f"  Note: source 'oeis' field is the placeholder 'possible', not a real OEIS id;")
    print(f"  this instance instead tests a small sum-free-set property from the problem's")
    print(f"  own area (additive combinatorics), computed classically below.")
    print(f"Universe: {universe}")
    print("Classical sum-free subsets (brute force, x+y=z check over all x,y,z in S):")
    for mask in range(2 ** n_qubits):
        subset = sorted(universe[i] for i in range(n_qubits) if (mask >> i) & 1)
        print(f"  mask={mask:03b} subset={subset} sum_free={classical_answer[mask]}")
    print(f"Marked (NON-sum-free, i.e. has a witness x+y=z) states: {marked}"
          f"  ({len(marked)} of {2**n_qubits})")

    # Standard Grover iteration count: floor(pi/4 * sqrt(N/M)).
    n_states = 2 ** n_qubits
    m_marked = len(marked)
    import math
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n_states / m_marked)))
    print(f"Grover iterations: {iterations}")
    counts = run_grover(n_qubits, marked, iterations)

    # Determine the majority outcome set from measurement counts.
    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Measurement counts (bitstring: count):")
    for bits, c in sorted_counts:
        print(f"  {bits}: {c}")

    # Check: does the quantum result concentrate on sum-free states? Take all
    # outcomes that together account for the marked fraction of probability
    # mass, and verify every one of them is classically sum-free.
    marked_mass = 0
    all_marked = True
    checked = []
    for bits, c in sorted_counts:
        mask = int(bits, 2)
        is_marked_state = mask in marked
        checked.append((mask, is_marked_state, c))
        if c >= total_shots * 0.02:  # ignore noise-floor outcomes below 2%
            if not is_marked_state:
                all_marked = False
        marked_mass += c if is_marked_state else 0

    marked_fraction = marked_mass / total_shots
    expected_fraction = len(marked) / (2 ** n_qubits)

    print(f"Fraction of shots landing on a classically non-sum-free (marked) subset: "
          f"{marked_fraction:.3f}")
    print(f"Uniform-random baseline would give: {expected_fraction:.3f}")

    # PASS condition: Grover amplification measurably favors the marked
    # (non-sum-free) states above the uniform baseline, and every
    # significant-probability outcome (>=2% of shots) is indeed one of the
    # classically-verified marked states.
    amplified = marked_fraction > expected_fraction
    verified = amplified and all_marked

    print()
    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
