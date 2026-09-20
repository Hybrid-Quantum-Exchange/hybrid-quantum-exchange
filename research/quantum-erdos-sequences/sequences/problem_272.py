"""
Erdos problem #272 (additive combinatorics / arithmetic progressions).

Source metadata (data/problems.yaml in the erdosproblems repo, entry
`number: "272"`):
    prize: no
    tags: ["additive combinatorics", "arithmetic progressions"]
    oeis: ["possible"]

NOTE ON OEIS ID: the "oeis" field for problem 272 is the literal string
"possible" -- not an actual OEIS sequence identifier -- and the repository
has no per-problem description page for #272 (no data/272.md / *.tex),
only the metadata block above. So there is no real OEIS id to anchor a
"membership in the sequence" style test to, and no informal statement text
to derive an exact finite instance of the actual open problem from. This
script is therefore an honest best-effort construction, not a literal
encoding of Erdos problem 272 itself: it builds a small, genuinely
computable instance in the same mathematical territory named by the tags
(existence of a 3-term arithmetic progression inside a finite subset of
integers), and tests a real Grover search circuit against it. Treat the
"OEIS id used" as: NONE (no valid id available for #272).

Classical property tested
--------------------------
Fix N = 8 and the subset (mod N):
    S = {0, 1, 2, 4, 5, 7}      (i.e. NOT in S: {3, 6})

Define the search space as x in Z_8 (3 qubits). x is a SOLUTION iff the
length-3 arithmetic progression with common difference 1, starting at x
and taken mod N -- {x, x+1 mod N, x+2 mod N} -- is entirely contained in S.

This is a finite, small, directly computable instance of "does a given
finite integer set contain a 3-term arithmetic progression starting at a
given point" -- exactly the kind of question the "arithmetic progressions"
/ "additive combinatorics" tags name, restricted to d = 1 and cyclic
wraparound so it fits in 3 qubits.

The script:
  1. Computes the classical solution set for x in {0,...,7} by brute force.
  2. Builds a Grover search circuit (oracle + diffuser) over 3 qubits that
     marks exactly those x.
  3. Runs it on the ideal AerSimulator and checks that the most-probable
     measured outcomes are exactly the classical solution set.
  4. Prints PASS/FAIL.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate

N = 8  # 2**3, so 3 qubits index x in Z_8
S = {0, 1, 2, 4, 5, 7}  # subset of Z_8; NOT in S: 3, 6


def is_solution(x: int) -> bool:
    """x is a solution iff {x, x+1 mod N, x+2 mod N} subset S."""
    return all(((x + k) % N) in S for k in range(3))


def classical_solution_set():
    return sorted(x for x in range(N) if is_solution(x))


def build_oracle(n_qubits: int, solutions: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each solution's computational basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for sol in solutions:
        bits = format(sol, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, solutions: list[int], shots: int = 4096):
    import math

    M = len(solutions)
    N_total = 2 ** n_qubits
    if M == 0 or M == N_total:
        # Degenerate cases: Grover isn't meaningful; handled by caller.
        return None

    # Optimal number of Grover iterations for M marked out of N_total.
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(N_total / M)))

    oracle = build_oracle(n_qubits, solutions)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 3
    solutions = classical_solution_set()
    print(f"N = {N}, S = {sorted(S)}")
    print(f"Classical solution set (x with {{x,x+1,x+2}} mod {N} subset S): {solutions}")

    out = run_grover(n_qubits, solutions)
    if out is None:
        print("Degenerate solution set; Grover search not applicable.")
        print("FAIL")
        return

    counts, iterations = out
    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")

    shots = sum(counts.values())
    # Convert bitstrings (Qiskit: qubit 0 is rightmost char) back to integers.
    measured_probs = {}
    for bitstring, c in counts.items():
        x = int(bitstring[::-1], 2)
        measured_probs[x] = measured_probs.get(x, 0) + c / shots

    # Take the top-M most frequently measured outcomes as the quantum answer.
    M = len(solutions)
    top_x = sorted(measured_probs, key=lambda k: -measured_probs[k])[:M]
    quantum_answer = sorted(top_x)

    # Sanity: the total probability mass on true solutions should be large.
    mass_on_solutions = sum(measured_probs.get(x, 0.0) for x in solutions)
    print(f"Quantum answer (top-{M} measured states): {quantum_answer}")
    print(f"Probability mass on classical solutions: {mass_on_solutions:.4f}")

    verified = (quantum_answer == solutions) and (mass_on_solutions > 0.8)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
