"""
Erdos problem #947 (per erdosproblems.com / the manman4/erdosproblems data
export, data/problems.yaml): tags ["number theory", "covering systems"],
oeis: ["N/A"]. The entry carries no OEIS sequence id at all -- so there is no
literal sequence to look up or to fabricate a value from. Per the task
instructions for that case, this script makes its best honest attempt at a
genuine small, finite, computable property drawn from the problem's own
subject matter (covering systems of congruences), and reports the limitation
plainly: this is a property motivated by the "covering systems" tag, not a
verification of problem #947's actual (unformalized-for-search-purposes)
statement, and not tied to any OEIS id, because none exists in the data.

Chosen property
----------------
A covering system is a finite set of congruences x = r_i (mod m_i) whose
union covers every integer. The atomic building block of any covering
system is single-congruence membership: "is x = r (mod m)?". That is a
small, finite, exactly-computable predicate, and it is the natural minimal
quantum-searchable object connected to the "covering systems" tag.

Concretely, for modulus m = 3, residue r = 1, and search space
x in {0, 1, ..., 7} (N = 8 = 2^3, so 3 qubits), the classical solution set is

    S = { x in [0, 8) : x mod 3 == 1 } = {1, 4, 7}

computed below in Python from first principles (no hard-coded literal).

Quantum approach
-----------------
Grover's algorithm on 3 qubits with a phase oracle that flags exactly the
x in S, followed by the optimal number of Grover diffusion iterations for
|S| = 3 out of N = 8 (1 iteration, per floor(pi/4 * sqrt(N/|S|))). The
circuit is run on the ideal AerSimulator (statevector-based sampling), and
the most frequent measured bitstring(s) are compared against the classical
set S. PASS if the top measurement outcome(s) (i.e. those with combined
plurality of shots) all lie in S.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solution_set(modulus: int, residue: int, n: int) -> set[int]:
    """x in [0, n) with x % modulus == residue, computed directly."""
    return {x for x in range(n) if x % modulus == residue}


def build_oracle(num_qubits: int, solutions: set[int]) -> QuantumCircuit:
    """Phase oracle: flips the sign of amplitude for each marked basis state."""
    oracle = QuantumCircuit(num_qubits, name="oracle")
    for target in solutions:
        bits = format(target, f"0{num_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            oracle.x(zero_positions)
        if num_qubits == 1:
            oracle.z(0)
        else:
            oracle.h(num_qubits - 1)
            oracle.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            oracle.h(num_qubits - 1)
        if zero_positions:
            oracle.x(zero_positions)
    return oracle


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    diffuser = QuantumCircuit(num_qubits, name="diffuser")
    diffuser.h(range(num_qubits))
    diffuser.x(range(num_qubits))
    diffuser.h(num_qubits - 1)
    if num_qubits == 1:
        diffuser.z(0)
    else:
        diffuser.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    diffuser.h(num_qubits - 1)
    diffuser.x(range(num_qubits))
    diffuser.h(range(num_qubits))
    return diffuser


def run_grover(modulus: int, residue: int, n: int, shots: int = 4096):
    num_qubits = n.bit_length() - 1
    assert 2 ** num_qubits == n, "n must be a power of two"

    solutions = classical_solution_set(modulus, residue, n)
    assert solutions, "need at least one solution for a meaningful Grover search"

    num_iterations = max(1, round(math.floor(
        (math.pi / 4) * math.sqrt(n / len(solutions))
    )))

    oracle = build_oracle(num_qubits, solutions)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(num_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # Convert little-endian measured bitstrings back to integers.
    int_counts: dict[int, int] = {}
    for bitstring, count in counts.items():
        value = int(bitstring[::-1], 2)
        int_counts[value] = int_counts.get(value, 0) + count

    return solutions, int_counts, num_iterations


def main() -> None:
    modulus, residue, n = 3, 1, 8

    solutions, int_counts, num_iterations = run_grover(modulus, residue, n)

    total_shots = sum(int_counts.values())
    # Plurality: the outcome(s) tied for the single highest shot count.
    max_count = max(int_counts.values())
    top_outcomes = sorted(v for v, c in int_counts.items() if c == max_count)

    solution_shots = sum(c for v, c in int_counts.items() if v in solutions)
    solution_fraction = solution_shots / total_shots

    print("Erdos problem #947 -- covering-systems-tagged sequence check")
    print(f"OEIS id(s) in source data: N/A (none provided)")
    print(f"Property tested: membership in {{x in [0,{n}) : x mod {modulus} == {residue}}}")
    print(f"Classical solution set S = {sorted(solutions)}")
    print(f"Grover iterations used: {num_iterations}")
    print(f"Top measured outcome(s) (plurality of {total_shots} shots): {top_outcomes}")
    print(f"Fraction of shots landing in S: {solution_fraction:.3f}")

    all_top_in_solutions = all(v in solutions for v in top_outcomes)
    amplified = solution_fraction > (len(solutions) / n)  # better than uniform baseline

    passed = all_top_in_solutions and amplified
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
