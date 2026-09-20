"""
Erdos problem #119 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems repo, entry "number: 119"):
    prize: $100
    status: solved (Lean), last update 2026-08-23
    oeis: ["N/A"]
    tags: ["analysis", "polynomials"]

LIMITATION, stated honestly up front: problem #119 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no specific
integer sequence to build a membership/term-search circuit *for*. Per the
task's fallback instruction, this script makes its best honest attempt at a
genuine, finite, computable quantum circuit in the spirit of the problem's
own tags ("analysis", "polynomials") rather than fabricating an OEIS-derived
property that does not exist for this entry.

Chosen property (self-contained, no OEIS dependency):
    Let p(x) = x^2 - 5x + 6 over the finite domain x in {0, 1, ..., 7}
    (3 bits). p is an integer-coefficient polynomial -- squarely in the
    "polynomials" tag. The classical property under test is:

        S = { x in {0,...,7} : p(x) == 0 }

    which is computed here from first principles by direct classical
    evaluation of p at every point in the domain (no hard-coded factoring,
    no OEIS lookup). For this p, S = {2, 3}, since p factors as
    (x-2)(x-3), but the script does not assume that -- it evaluates p(x)
    at all 8 points and collects the zeros.

Quantum method: Grover's search circuit (3 qubits, real oracle + diffuser,
run on the ideal AerSimulator) is used to search the domain for the
elements of S. The oracle is synthesized as a diagonal phase-flip unitary
constructed directly from the classically-computed marked set S (a standard
"black-box marks known solutions" Grover oracle), then amplitude
amplification and measurement are executed for real on the simulator. The
test compares the *measured* most-probable outcomes against the
*classically computed* S.

PASS criterion: the two basis states measured with highest probability
after Grover iterations are exactly the two elements of S (i.e. Grover's
algorithm quantum-mechanically finds the same roots the classical
evaluation found).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator


def p(x: int) -> int:
    """The polynomial under test: p(x) = x^2 - 5x + 6."""
    return x * x - 5 * x + 6


def classical_roots(domain_bits: int) -> list[int]:
    """Evaluate p at every point of the domain {0,...,2**domain_bits - 1}
    and return the classical zero set, from first principles."""
    n = 2 ** domain_bits
    return [x for x in range(n) if p(x) == 0]


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Diagonal phase-flip oracle: multiplies each marked basis state's
    amplitude by -1, leaves all others untouched. Built directly from the
    classically-computed marked set (standard Grover oracle synthesis for a
    black-box classical predicate)."""
    dim = 2 ** n_qubits
    diag = np.ones(dim, dtype=complex)
    for m in marked:
        diag[m] = -1.0
    qc = QuantumCircuit(n_qubits, name="oracle")
    qc.unitary(Operator(np.diag(diag)), range(n_qubits), label="phase_flip")
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    domain_bits = 3  # domain {0,...,7}
    n = 2 ** domain_bits

    # --- classical computation, first principles ---
    marked = classical_roots(domain_bits)
    print(f"Classical evaluation of p(x) = x^2 - 5x + 6 over x in 0..{n - 1}:")
    for x in range(n):
        print(f"  p({x}) = {p(x)}")
    print(f"Classical root set S = {marked}")

    if not marked:
        print("No roots in domain; nothing to search for. FAIL")
        return False

    # --- Grover search circuit ---
    num_solutions = len(marked)
    # optimal number of Grover iterations for this search space/solution count
    iterations = max(1, math.floor((math.pi / 4) * math.sqrt(n / num_solutions)))

    oracle = build_oracle(domain_bits, marked)
    diffuser = build_diffuser(domain_bits)

    qc = QuantumCircuit(domain_bits, domain_bits)
    qc.h(range(domain_bits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(domain_bits))
        qc.append(diffuser.to_instruction(), range(domain_bits))
    qc.measure(range(domain_bits), range(domain_bits))

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim)
    job = sim.run(tqc, shots=shots)
    counts = job.result().get_counts()

    # top-|marked| most frequent measured outcomes (bitstrings -> ints, qiskit
    # is little-endian in the classical register string, c[n-1]...c[0])
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_counts[:num_solutions]
    measured_top = sorted(int(bits, 2) for bits, _ in top_k)

    print(f"Grover iterations used: {iterations}")
    print(f"Measured counts: {counts}")
    print(f"Top-{num_solutions} measured basis states: {measured_top}")
    print(f"Classical root set:                 {sorted(marked)}")

    verified = measured_top == sorted(marked)
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
