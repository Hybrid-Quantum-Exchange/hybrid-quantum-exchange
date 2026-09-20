"""
Erdos problem #517 -- quantum-testable-sequence lane (attempted, limited).

Source record (erdosproblems.com data, entry "number: \"517\"" in
data/problems.yaml as cloned at /home/user/manman4/erdosproblems):

    number: "517"
    prize: "no"
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["analysis"]

HONEST LIMITATION (read before trusting PASS/FAIL below)
----------------------------------------------------------------
Problem #517 has NO associated OEIS sequence (oeis: ["N/A"]) and is an
*open* problem in analysis with no known finite characterization. There is
therefore no real integer sequence here to build a genuine, mathematically
faithful quantum-testable property from -- the task's own instructions say
that in this situation the honest move is to write the script anyway, note
the gap plainly, and not fabricate a property or fake a pass tied to
problem 517 specifically.

What this script actually does instead: it is a generic, self-contained
Grover's-algorithm search circuit over a small finite space, run on the
ideal AerSimulator, whose classical answer is computed from first
principles in this same script and cross-checked against the quantum
result. This demonstrates a genuinely working small quantum circuit
(so the PASS/FAIL below is a real, non-fabricated verification of THAT
circuit), but it is NOT a test of any property of Erdos problem #517's
sequence, because problem #517 has no sequence to test.

Concretely: Grover search over the 4-qubit space {0,...,15} for the unique
marked element x0 = 11 (binary 1011), verified against the classical
answer (found by direct/brute-force scan, independent of the oracle
construction) that x0 is indeed the unique integer in [0,16) equal to 11.

verified_against_classical == True means: the quantum circuit's measured
result matches, with correct answer having the largest observed
probability, the classical/brute-force computation *of the small demo
instance above* -- not a verified property of problem 517's own sequence,
which does not exist.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS          # search space size: 16
MARKED = 11                 # the "answer" the oracle marks (binary 1011)


def classical_answer(n: int, marked: int) -> int:
    """Brute-force scan of the search space, independent of the oracle
    circuit construction below -- the classical ground truth."""
    found = [x for x in range(n) if x == marked]
    assert len(found) == 1
    return found[0]


def build_oracle(n_qubits: int, marked: int) -> QuantumCircuit:
    """Phase oracle: flips the sign of |marked> and leaves all other
    basis states unchanged, via a multi-controlled Z conditioned on the
    bit pattern of `marked` (X-sandwich the 0-bits, MCZ, undo)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(marked, f"0{n_qubits}b")[::-1]  # little-endian per qubit
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for i in zero_positions:
        qc.x(i)
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


def build_grover_circuit(n_qubits: int, marked: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run() -> None:
    truth = classical_answer(N, MARKED)

    # Optimal number of Grover iterations for one marked item in N.
    iterations = max(1, round((np.pi / 4) * np.sqrt(N)))

    circuit = build_grover_circuit(N_QUBITS, MARKED, iterations)

    backend = AerSimulator()
    shots = 4096
    result = backend.run(circuit, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints the classical register bitstring with clbit n-1 on the
    # left (standard big-endian display), which already equals the
    # integer index consistent with build_oracle's little-endian qubit
    # indexing (qubit i has weight 2**i) -- no reversal needed.
    def bitstring_to_int(bs: str) -> int:
        return int(bs, 2)

    best_bitstring = max(counts, key=counts.get)
    quantum_answer = bitstring_to_int(best_bitstring)
    quantum_prob = counts[best_bitstring] / shots

    print(f"Erdos problem #517: oeis=['N/A'], tags=['analysis'], status=open")
    print("No OEIS sequence exists for this problem; see module docstring "
          "for the honest limitation -- this circuit is a generic Grover "
          "demo, not a test of problem #517's (nonexistent) sequence.")
    print(f"Classical brute-force answer: {truth}")
    print(f"Quantum (Grover, {iterations} iterations, {shots} shots) "
          f"most-likely answer: {quantum_answer} (p={quantum_prob:.3f})")

    verified = (quantum_answer == truth) and (quantum_prob > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    run()
