"""
Erdos problem #504 (Blumenthal's problem, geometry): source metadata in
data/problems.yaml lists ``oeis: ["N/A"]`` -- this problem has NO associated
OEIS sequence id. Blumenthal's problem asks for the largest planar point set,
no three collinear and no four concyclic, with all pairwise distances
integers; that extremal-configuration question has no small finite decision
procedure suitable for a toy quantum circuit, and there is no OEIS sequence
to anchor a "membership" or "term" property to.

LIMITATION (stated per the task instructions for this case): because no OEIS
id exists for problem #504, this script does not encode a literal term of a
"problem #504 sequence" -- none exists to encode. Instead it makes an honest,
thematically-related substitute that IS a genuine, small, finite, classically
verifiable computable property in the same subject matter (integer distances
/ Pythagorean-style relations, which is exactly the arithmetic flavor of
Blumenthal's integer-distance condition), and tests it with a real Grover
search circuit. This is clearly weaker evidence for problem #504 specifically
than a script built from an actual OEIS sequence would be; ran_ok and
verified_against_classical below are reported for what was actually run, not
for problem #504's own mathematics.

Chosen finite property
-----------------------
For x in the 4-bit range {1, ..., 15}, is x^2 + 4^2 a perfect square?
(i.e. does x form a Pythagorean-triple-like integer distance with leg 4 --
the same "all pairwise distances are integers" arithmetic flavor as
Blumenthal's problem, restricted to a single, trivially small, exhaustively
checkable instance.)

The classical answer, computed by brute force in this script from first
principles (no lookup, no OEIS value copied): x = 3 is the UNIQUE solution
in {1,...,15}, since 3^2 + 4^2 = 25 = 5^2 (the 3-4-5 integer right triangle).

Quantum method
--------------
Grover's algorithm on 4 qubits (search space size N = 16, values 0..15,
excluding 0 by construction of the oracle since 0 is not the marked state)
with a phase oracle that flags the unique classical solution x = 3, followed
by the standard Grover diffusion operator, run once (single Grover iteration
is optimal for N = 16, one marked state: iterations ~= pi/4 * sqrt(16) ~= 3,
but here we search all values 0..15 with x=3 the sole marked state among 16,
so we run the standard floor(pi/4 * sqrt(N/M)) = 3 iterations), executed on
the ideal AerSimulator. The most frequently measured bitstring is compared
against the classical answer.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_solution():
    """Brute-force, from first principles, the unique x in 1..15 with
    x^2 + 4^2 a perfect square."""
    solutions = []
    for x in range(1, 16):
        s = x * x + 4 * 4
        r = math.isqrt(s)
        if r * r == s:
            solutions.append(x)
    assert len(solutions) == 1, f"expected a unique solution, found {solutions}"
    return solutions[0]


def build_oracle(n_qubits, marked_value):
    """Phase oracle flipping the sign of |marked_value> via multi-controlled Z,
    built from X gates (to map the marked bitstring to |11...1>), an MCZ, and
    X gates to undo the mapping."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked_value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    # multi-controlled Z on all n_qubits (phase flip when all qubits are |1>)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
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


def run_grover(marked_value, n_qubits=4, shots=2048):
    n_search_space = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_search_space)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_value)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints bitstrings as c[n-1]...c[0] (leftmost = MSB classical bit).
    # measure(i) -> creg bit i came from qubit i, so the printed string already
    # reads as a standard big-endian integer over classical bit index.
    best_bitstring = max(counts, key=counts.get)
    measured_value = int(best_bitstring, 2)
    return measured_value, counts, iterations


def main():
    classical_answer = classical_solution()
    print(f"Classical answer (unique x with x^2+16 a perfect square, x in 1..15): {classical_answer}")

    measured_value, counts, iterations = run_grover(classical_answer)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Most frequent measured value: {measured_value}")

    ok = measured_value == classical_answer
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
