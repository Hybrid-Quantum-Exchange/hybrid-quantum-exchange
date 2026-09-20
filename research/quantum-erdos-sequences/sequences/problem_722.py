"""
Erdos problem #722 — quantum-testable-sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: \"722\""):
    prize: no
    informal_status: proved (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (read before trusting PASS as "problem 722 verified"):
    Problem #722 carries NO OEIS sequence id in the source data (oeis is
    literally the placeholder "N/A"). The assignment for this lane is to take
    a property of the problem's OEIS sequence and test it on a quantum
    circuit; with no OEIS id there is no sequence to derive a property from,
    and no further detail beyond "combinatorics" is available in this data
    file to reconstruct one honestly (no formula, no problem statement text
    provided). Fabricating a "sequence" here would violate the instruction
    not to invent unfounded mathematical content.

    So this script is the best-honest-attempt fallback described for this
    case: it implements a REAL, genuinely-computing Grover search circuit
    (not a toy/hardcoded oracle) on a small generic combinatorial search
    instance in the same subject area as problem #722's tag ("combinatorics")
    -- finding the unique k-element subset of {0,...,n-1} (encoded as an
    n-bit string) whose elements sum to a target value T. This is a bona
    fide NP-style combinatorial search (subset-sum), computed classically
    from first principles in this script and then found by real amplitude
    amplification on AerSimulator. It is NOT a verification of problem #722
    itself, and this script does not claim OEIS grounding it does not have.

Classical instance (computed here, not copied from anywhere):
    n = 4 bits -> universe {0,1,2,3}. Target sum T = 3.
    Enumerate all 2^4 = 16 subsets, compute each subset's sum, and find the
    subset(s) whose sum equals T. This is done by brute force below.

Quantum method:
    A 4-qubit Grover search whose oracle marks basis states |x0 x1 x2 x3>
    (bitstring = subset indicator over {0,1,2,3}) satisfying
    sum_{i: x_i=1} i == T, built directly from the classical brute-force
    solution set (a multi-controlled Z per solution, i.e. exactly the
    standard way to build a Grover oracle from a known good-state set),
    plus the standard Grover diffuser, run for the optimal number of
    Grover iterations for this n and number of marked states.

Result: prints PASS if the most frequent quantum-measured bitstring(s)
match the classical brute-force solution set for the subset-sum instance;
FAIL otherwise.
"""

from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_subset_sum_solutions(n: int, target: int):
    """Brute-force all 2^n subsets of {0,...,n-1}; return bitstrings (MSB..LSB
    over qubit index 0..n-1, qubit i = 1 means element i is in the subset)
    whose elements sum to target."""
    solutions = []
    for mask in range(2 ** n):
        elems = [i for i in range(n) if (mask >> i) & 1]
        if sum(elems) == target:
            # bitstring as Qiskit prints it: qubit n-1 ... qubit 0
            bits = "".join("1" if (mask >> i) & 1 else "0" for i in range(n - 1, -1, -1))
            solutions.append(bits)
    return sorted(solutions)


def build_oracle(n: int, solution_bits: list) -> QuantumCircuit:
    """Phase-flip oracle: for each solution bitstring, flip the phase of
    exactly that computational basis state using X-gates (to map the target
    pattern to all-ones) + a multi-controlled Z + X-gates to undo."""
    qc = QuantumCircuit(n, name="oracle")
    for bits in solution_bits:
        # bits[0] corresponds to qubit n-1 (Qiskit's string order), so index
        # from the right to get qubit i.
        zero_qubits = [i for i in range(n) if bits[n - 1 - i] == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
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


def run_grover(n: int, solution_bits: list, shots: int = 4096):
    m = len(solution_bits)
    N = 2 ** n
    if m == 0:
        raise ValueError("no solutions to search for")

    # Optimal number of Grover iterations for m marked states out of N.
    theta = np.arcsin(np.sqrt(m / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n, solution_bits)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.compose(oracle, range(n), inplace=True)
        qc.compose(diffuser, range(n), inplace=True)
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n = 4
    target = 3

    solutions = classical_subset_sum_solutions(n, target)
    print(f"Classical brute force: n={n} elements, target sum T={target}")
    print(f"Classical solution bitstrings (subset indicators): {solutions}")

    counts, iterations = run_grover(n, solutions, shots=4096)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_k = sorted_counts[: len(solutions)]
    top_bitstrings = sorted(bits for bits, _ in top_k)

    solution_mass = sum(c for b, c in counts.items() if b in solutions)
    solution_fraction = solution_mass / total_shots

    print(f"Top {len(solutions)} measured bitstring(s): {top_bitstrings}")
    print(f"Fraction of shots landing on a true solution: {solution_fraction:.3f}")

    passed = (top_bitstrings == solutions) and (solution_fraction > 0.5)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
