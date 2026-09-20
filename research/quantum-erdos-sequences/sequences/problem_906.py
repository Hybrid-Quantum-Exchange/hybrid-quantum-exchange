"""
Erdos problem #906 -- quantum-testable-sequence attempt (HONEST NON-PASS).

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"906\""):

    number: "906"
    prize: "no"
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis", "iterated functions"]

There is no OEIS sequence attached to problem #906 (oeis == ["N/A"]). The
problem's tags ("analysis", "iterated functions") indicate it concerns the
asymptotic/limiting behaviour of an iterated real- or complex-valued function
-- the kind of statement erdosproblems.com states over the reals/integers
without reducing to a single finite integer sequence in OEIS. Per the task
instructions: do not fabricate a "small computable property" with no real
mathematical content, and do not invent a property when there is no
OEIS-backed sequence to derive it from.

Because there is no OEIS id for this problem, this script does NOT construct
a genuine finite instance of the *actual* Erdos #906 statement. Building one
would require guessing at the analytic statement from the tags alone, which
is exactly the fabrication the task instructions rule out.

What this script does instead, to be maximally useful while staying honest:
it runs a real, correctly-verified quantum circuit (Grover search implemented
with genuine oracle + diffusion operators on the AerSimulator) against a
*generic* small finite search instance -- "find the unique x in {0,...,7}
such that x == 5" -- purely as a working-infrastructure smoke test. This
circuit is real and its own classical answer is verified from first
principles in code below. It is explicitly NOT a computation of any property
of Erdos problem #906, and the script reports that fact rather than claiming
otherwise.

Result semantics:
  - ran_ok        : True if the script executes without error.
  - verified_against_classical : False, by design/reporting convention here,
    because no OEIS-backed classical property of problem #906 exists to
    verify a quantum result against. The Grover demo below is internally
    self-consistent (its own classical target matches its own quantum
    output) but that is a generic infrastructure check, not a verification
    of anything about problem #906.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_search(space, predicate):
    """Brute-force classical search over `space`, from first principles."""
    hits = [x for x in space if predicate(x)]
    return hits


def build_grover_circuit(n_qubits: int, marked: int) -> QuantumCircuit:
    """Real Grover search circuit marking the computational basis state
    `marked` (an integer in [0, 2**n_qubits)) via a genuine multi-controlled-Z
    oracle, with the standard diffusion (inversion-about-mean) operator.
    Iteration count uses the standard Grover formula floor(pi/4 * sqrt(N)).
    """
    N = 2 ** n_qubits
    qc = QuantumCircuit(n_qubits, n_qubits)

    # uniform superposition
    qc.h(range(n_qubits))

    def oracle(qc: QuantumCircuit):
        bits = format(marked, f"0{n_qubits}b")[::-1]
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(i)

    def diffuser(qc: QuantumCircuit):
        qc.h(range(n_qubits))
        qc.x(range(n_qubits))
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.x(range(n_qubits))
        qc.h(range(n_qubits))

    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N))))
    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    print(__doc__)

    # -- Erdos #906 itself: no OEIS id, no finite computable property derived. --
    problem_number = "906"
    oeis_ids = ["N/A"]
    tags = ["analysis", "iterated functions"]
    print(f"Erdos problem #{problem_number}: oeis={oeis_ids}, tags={tags}")
    print("No OEIS sequence is attached to this problem; no genuine finite")
    print("computable property of an OEIS sequence was derived for it.")

    # -- Generic infrastructure smoke test (NOT a test of problem #906) --
    n_qubits = 3  # search space size N = 8 <= 64, as required
    space = list(range(2 ** n_qubits))
    target = 5

    classical_hits = classical_search(space, lambda x: x == target)
    assert classical_hits == [target], "classical brute-force sanity check failed"
    classical_answer = classical_hits[0]
    print(f"\n[Infrastructure smoke test only] classical search over {space} "
          f"for x == {target}: answer = {classical_answer}")

    qc = build_grover_circuit(n_qubits, target)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=2048).result()
    counts = result.get_counts()
    most_likely = max(counts, key=counts.get)
    quantum_answer = int(most_likely[::-1], 2)  # qiskit bit ordering

    smoke_test_pass = (quantum_answer == classical_answer)
    print(f"Grover circuit measured most-likely state = {quantum_answer} "
          f"(counts: {counts})")
    print("Infrastructure smoke test:", "PASS" if smoke_test_pass else "FAIL")

    print("\nOverall verdict for Erdos problem #906: FAIL")
    print("Reason: no OEIS sequence exists for problem #906, so no genuine")
    print("quantum-testable classical property of it could be verified here.")


if __name__ == "__main__":
    main()
