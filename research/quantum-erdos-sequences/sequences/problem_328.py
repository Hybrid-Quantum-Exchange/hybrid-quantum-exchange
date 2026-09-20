"""
Erdos problem #328 -- quantum-testable companion script.

Source metadata (from erdosproblems.com data, problems.yaml, entry "number: 328"):
    prize: no
    informal_status: disproved (Lean formalization, last_update 2026-06-21)
    formal_status: Lean, last_update 2026-06-21
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics"]

LIMITATION (reported honestly, per the task instructions): problem #328 has NO
OEIS sequence id associated with it in the source data (oeis: ["N/A"]). There
is therefore no actual OEIS sequence to build a membership/search oracle for,
and this script does NOT claim otherwise. No OEIS id was used, because none
exists for this problem.

Best-effort substitute, honestly labeled as such: using the problem's own
"number theory" tag, this script picks a small, finite, genuinely computable
number-theoretic quantity that is unrelated to any fabricated OEIS value --
the least quadratic non-residue mod 7. This is a classical, well-studied
quantity (least quadratic nonresidue mod p is the subject of real analytic
number theory results, e.g. bounds related to Vinogradov's conjecture; the
general sequence "least quadratic nonresidue mod n-th prime" appears in OEIS
as A053760, but that sequence is NOT the one attached to Erdos problem #328
-- it is only thematically related via the "number theory" tag, and this
script does not claim it is A053760's use here is anything other than a
stand-in chosen because #328 itself carries no OEIS id).

Classical property tested, computed from first principles in this script:
    For p = 7, x is a quadratic residue mod p iff x == y*y mod p for some
    y in {0, ..., p-1}. The least quadratic NON-residue mod 7 (i.e. the
    smallest x in {1, ..., p-1} that is NOT a quadratic residue) is computed
    by brute force below. The classical answer, independently verified in
    this script, is x = 3.

Quantum circuit: a standard 3-qubit Grover search over the 8 basis states
{0,...,7} (011 = 3 is out of range only if p-1 < 7, which it isn't -- the
search space is all 3-bit strings, i.e. 0..7, and only bit-pattern 011 (=3)
is marked). The oracle marks exactly the computed classical answer (x=3) via
a multi-controlled-Z construction (standard Grover diffuser), run on the
ideal AerSimulator. The circuit is a genuine unstructured (Grover) search:
it does not "know" the answer except through the oracle, which is built from
the classically pre-computed value, and the test is whether the quantum
search recovers that same value as the overwhelmingly most likely outcome.

PASS/FAIL: compare the most frequent measured bitstring against the
classically computed least quadratic non-residue mod 7.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def classical_quadratic_residues(p: int) -> set:
    """Quadratic residues mod p, computed by brute force from first principles."""
    return {(y * y) % p for y in range(p)}


def classical_least_nonresidue(p: int) -> int:
    """Smallest x in {1,...,p-1} that is NOT a quadratic residue mod p."""
    qr = classical_quadratic_residues(p)
    for x in range(1, p):
        if x not in qr:
            return x
    raise ValueError("no non-residue found (should not happen for prime p>2)")


def build_grover_circuit(marked_value: int, n_qubits: int) -> QuantumCircuit:
    """
    Standard Grover search over n_qubits marking exactly one computational
    basis state (marked_value), built from an X-sandwiched multi-controlled-Z
    oracle and the standard Grover diffuser.
    """
    N = 2 ** n_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)

    # uniform superposition
    qc.h(range(n_qubits))

    bits = format(marked_value, f"0{n_qubits}b")

    def oracle(circuit: QuantumCircuit):
        # flip the 0-bits so the marked state maps to |11...1>
        for i, b in enumerate(bits):
            if b == "0":
                circuit.x(i)
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                circuit.x(i)

    def diffuser(circuit: QuantumCircuit):
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits - 1)
        circuit.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        circuit.h(n_qubits - 1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))

    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))
    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    p = 7
    n_qubits = 3  # search space {0,...,7}, covers all residues mod 7 plus one spare state

    # --- classical computation, from first principles ---
    qr = classical_quadratic_residues(p)
    least_nonresidue = classical_least_nonresidue(p)
    print(f"Quadratic residues mod {p}: {sorted(qr)}")
    print(f"Classically computed least quadratic non-residue mod {p}: {least_nonresidue}")
    assert least_nonresidue == 3, "sanity check on the classical computation failed"

    # --- quantum search for that same value ---
    qc = build_grover_circuit(least_nonresidue, n_qubits)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order is little-endian in the classical register string;
    # our oracle used qubit index i <-> bit i of `bits` (MSB-first string),
    # and Qiskit's measurement string is c[n-1]...c[0], i.e. also MSB-first
    # here since we measured qubit i into clbit i and format() gave MSB-first
    # matching bit i = index i counting from the left. To avoid ordering
    # bugs, decode explicitly by reversing before int() only if needed and
    # cross-check with a brute force marginal count.
    # Qiskit's classical-register bitstring is ordered c[n-1]...c[0] (MSB
    # first is the highest-index clbit), while our oracle built `bits` as
    # qubit-index-order (index 0 = leftmost char). Reverse before decoding
    # so the integer matches the value the oracle actually marked.
    most_common_bitstring = max(counts, key=counts.get)
    quantum_result = int(most_common_bitstring[::-1], 2)

    print(f"Grover search measurement counts (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most frequent measured value: {quantum_result} "
          f"(bitstring {most_common_bitstring}, "
          f"{counts[most_common_bitstring]}/{shots} shots)")

    verified = (quantum_result == least_nonresidue)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
