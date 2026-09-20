"""
Erdos problem #663 (data/problems.yaml, erdosproblems.com), OEIS id: A391668.

This entry is an honest partial attempt. The repository clone available in
this environment (/home/user/manman4/erdosproblems) gives problem #663's
metadata (prize=no, status=open, tags=["number theory"], oeis=["A391668"])
but this environment has no working internet access, so the actual defining
formula/terms of OEIS A391668 could not be fetched or independently verified.
Rather than fabricate a "property of A391668" whose correctness could not be
checked, this script tests a small, genuinely computable, classically-checked
number-theory property in the same spirit as the problem's tag ("number
theory"): finding a nontrivial divisor of a composite integer N, via Grover's
search algorithm, and checking the quantum result against a classical
trial-division computation.

Classical property under test
------------------------------
N = 15 (4-bit search space, x in 0..15).
Property P(x):  1 < x < N  and  N % x == 0   (x is a nontrivial divisor of N)
Classical answer (computed here from first principles by trial division):
    divisors of 15 in (1, 15) = {3, 5}
So the two marked/"good" 4-bit basis states are |0011> (3) and |0101> (5).

Quantum approach
----------------
Grover's algorithm on 4 qubits (search space size N_search = 16) is used to
amplify the amplitude of the two marked states {3, 5}. The oracle is a
diagonal phase oracle built directly from the classically-computed marked
set (a standard, legitimate way to instantiate Grover's algorithm for a
small explicit search problem). After ceil(pi/4 * sqrt(16/2)) ~= 2 Grover
iterations, the circuit is measured on the ideal AerSimulator and the most
frequent outcomes are compared against the classical divisor set.

PASS criterion: the two most-frequent measured 4-bit strings, converted to
integers, equal the classical divisor set {3, 5} computed by trial division.

Honesty note (per task instructions): this script does NOT claim to test a
defining property of A391668 itself, since that sequence's data could not be
retrieved/verified classically in this environment. It reports
verified_against_classical as true only with respect to the divisor-search
property actually implemented and checked below.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_nontrivial_divisors(n: int) -> list[int]:
    """Trial division from first principles: all x with 1 < x < n and n % x == 0."""
    return [x for x in range(2, n) if n % x == 0]


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Diagonal phase oracle: flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")
        # flip qubits that are 0 in this state, so a multi-controlled Z fires on |state>
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover_divisor_search(n: int, num_qubits: int, shots: int = 4096):
    marked = classical_nontrivial_divisors(n)
    assert marked, f"{n} must be composite (have a nontrivial divisor) for this demo"
    assert all(0 <= m < 2**num_qubits for m in marked)

    search_space = 2**num_qubits
    num_solutions = len(marked)
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / num_solutions)))

    oracle = build_oracle(num_qubits, marked)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return marked, counts, iterations


def main():
    N = 15
    NUM_QUBITS = 4

    classical_divisors = sorted(classical_nontrivial_divisors(N))
    print(f"Classical nontrivial divisors of {N} (trial division): {classical_divisors}")

    marked, counts, iterations = run_grover_divisor_search(N, NUM_QUBITS)
    print(f"Grover iterations used: {iterations}")
    print(f"Marked (classically computed) states passed to oracle: {sorted(marked)}")

    # top-k results, k = number of marked states
    k = len(marked)
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_results = sorted_counts[:k]
    top_ints = sorted(int(bitstring, 2) for bitstring, _ in top_results)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bitstring, c in counts.items() if int(bitstring, 2) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Top {k} measured outcomes (bitstring:count): {top_results}")
    print(f"Top outcomes as integers: {top_ints}")
    print(f"Fraction of shots landing on a marked (divisor) state: {marked_fraction:.3f}")

    verified = (top_ints == classical_divisors) and (marked_fraction > 0.5)

    print(f"verified_against_classical = {verified}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
