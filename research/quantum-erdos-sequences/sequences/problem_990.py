"""
Erdos problem #990 -- quantum-testable-sequences lane.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: \"990\""):
    prize: no
    informal_status: disproved (2026-04-19)
    formal_status: Lean (2026-04-19)
    oeis: ["N/A"]
    tags: ["analysis"]

LIMITATION (read before trusting PASS/FAIL below):
Problem #990 carries no OEIS sequence id -- its oeis field is literally
["N/A"] -- and its single tag is "analysis", not a combinatorial/number-
theoretic tag pointing at a countable object. There is therefore no genuine
integer sequence attached to this problem to build a small finite/computable
membership, divisibility, primality, or counting property from, as the task
requires. Fabricating one (e.g. inventing an unrelated OEIS id) would violate
the explicit instruction not to fabricate mathematical content.

Honest best-effort fallback actually implemented below:
Since no real sequence-derived property exists for #990, this script does
NOT claim to verify anything about problem #990's actual mathematical
content. Instead, as the fallback the task instructions explicitly allow
("write the script anyway with your best honest attempt"), it implements a
genuine, self-contained quantum computation -- Grover's search algorithm --
on a small finite/computable search problem that is at least in the spirit
of the "analysis" tag's flavor of extremal search: finding the unique integer
n in the range [0, N-1] (N = 16, so 4 qubits) satisfying a simple arithmetic
marking predicate (n is congruent to a fixed residue mod a fixed modulus,
picked so exactly one n in [0, N) satisfies it). The classical answer is
computed first, from scratch, by brute-force enumeration; the quantum result
(most frequently measured basis state after running Grover's algorithm on
the ideal AerSimulator) is then compared against it.

This confirms only that the Grover circuit correctly amplifies the marked
classical answer -- it is a generic quantum-search demonstration for lane
completeness, NOT a verification of any property specific to Erdos problem
#990's mathematical content, because #990 has no OEIS-indexed sequence to
tie such a property to.

Reporting (per the task's honesty requirement):
    ran_ok: whether this script runs to completion without error.
    verified_against_classical: whether the quantum measurement matches the
        classical brute-force answer for the fallback search problem above --
        NOT a claim of verifying problem #990's own content, which is not
        possible from the available metadata.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_answer(n_qubits: int, modulus: int, residue: int) -> int:
    """Brute-force, from first principles: the unique n in [0, 2**n_qubits)
    with n % modulus == residue. Raises if the marked set isn't a singleton,
    since Grover's algorithm as built below assumes exactly one marked
    state."""
    N = 2 ** n_qubits
    marked = [n for n in range(N) if n % modulus == residue]
    if len(marked) != 1:
        raise ValueError(
            f"expected exactly one marked state in [0,{N}), found {marked}"
        )
    return marked[0]


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state
    |target> (multi-controlled Z conditioned on the bit pattern of target)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian per qubit
    zero_qubits = [i for i, b in enumerate(bits) if b == "0"]

    for q in zero_qubits:
        qc.x(q)

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for q in zero_qubits:
        qc.x(q)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(n_qubits: int, target: int, iterations: int) -> QuantumCircuit:
    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main() -> bool:
    n_qubits = 4          # N = 16 search space
    modulus = 16
    residue = 11           # marked classical answer, derived below

    target = classical_answer(n_qubits, modulus, residue)
    print(f"Problem #990: no OEIS id available (oeis: ['N/A'], tags: ['analysis']).")
    print("Falling back to a generic Grover-search demonstration (see docstring).")
    print(f"Classical brute-force answer: unique n in [0,{2**n_qubits}) with "
          f"n % {modulus} == {residue} is n = {target}")

    N = 2 ** n_qubits
    optimal_iterations = max(1, round(np.pi / 4 * np.sqrt(N)))
    qc = build_grover_circuit(n_qubits, target, optimal_iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 2048
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    most_common_bitstring = max(counts, key=counts.get)
    quantum_answer = int(most_common_bitstring, 2)
    confidence = counts[most_common_bitstring] / shots

    print(f"Grover circuit: {n_qubits} qubits, {optimal_iterations} iteration(s), "
          f"{shots} shots")
    print(f"Most frequent measured state: {most_common_bitstring} = {quantum_answer} "
          f"(probability {confidence:.3f})")

    ok = (quantum_answer == target) and (confidence > 0.5)
    if ok:
        print("PASS: quantum Grover search result matches classical answer")
    else:
        print("FAIL: quantum Grover search result does not match classical answer")
    return ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
