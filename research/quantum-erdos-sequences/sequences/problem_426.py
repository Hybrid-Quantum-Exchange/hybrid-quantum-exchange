"""
Erdos problem #426 -- quantum-testable sequence attempt.

Source record (data/problems.yaml, erdosproblems repo, entry "number: '426'"):
    prize: $25
    informal_status: disproved (Lean-formalized 2026-04-20)
    tags: ["graph theory"]
    oeis: ["possible"]

LIMITATION (read before trusting anything below): the "oeis" field for this
problem is the literal string "possible" -- not an OEIS A-number. There is no
real OEIS sequence id attached to problem #426 in the source data, and the
problem's tag ("graph theory") gives no numeric sequence either -- it names a
subject area, not a computable integer sequence. Per the task instructions we
therefore do NOT fabricate a property or invent an oeis id/value to test:
there is nothing given here to derive a genuine "is x in the sequence" or
"n-th term" computation from.

What this script does instead, honestly:
It builds a real, small Grover search circuit (genuine amplitude
amplification, not a lookup table) for a toy but well-defined and CLASSICALLY
VERIFIED arithmetic property inspired by the problem's "graph theory" tag:
finding the unique 3-bit number n in [0, 7] such that n encodes a triangle-free
2-edge-colouring marker -- concretely, n such that popcount(n) == 2 AND n is
even (a simple, checkable predicate on 3-bit strings). This is NOT a term of
any OEIS sequence tied to problem #426; it is a placeholder oracle so the
circuit is a genuine, checkable Grover instance rather than a fabricated
"quantum test" of problem #426 itself.

Because this is not actually testing problem #426's mathematics (no OEIS id
exists to test), verified_against_classical for the *problem* is honestly
False. What IS verified is that the Grover circuit's measured answer matches
the classical brute-force answer to the toy predicate -- i.e. the quantum
mechanism itself runs and works, faithfully reported below.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 3  # search space {0,...,7}


def classical_predicate(n: int) -> bool:
    """popcount(n) == 2 and n is even, for n in [0, 7]."""
    return bin(n).count("1") == 2 and n % 2 == 0


def classical_answer():
    """Brute-force over the full 3-bit space; the true 'marked' set."""
    return [n for n in range(2 ** N_QUBITS) if classical_predicate(n)]


def build_oracle(marked, n_qubits):
    """Phase-flip oracle marking each n in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for n in marked:
        bits = format(n, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    marked = classical_answer()
    assert marked == [6], f"unexpected classical predicate result: {marked}"

    n_items = 2 ** N_QUBITS
    m = len(marked)
    # standard optimal Grover iteration count
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items / m)))

    qc = build_grover_circuit(marked, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 2048
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # convert bitstrings (qiskit little-endian, c-reg order) to ints
    int_counts = {}
    for bitstring, c in counts.items():
        n = int(bitstring, 2)
        int_counts[n] = int_counts.get(n, 0) + c

    measured_marked_prob = sum(int_counts.get(n, 0) for n in marked) / shots
    top_n = max(int_counts, key=int_counts.get)

    quantum_ok = (top_n in marked) and (measured_marked_prob > 0.8)

    print("Erdos problem #426 quantum-testable-sequence attempt")
    print("problems.yaml oeis field:", ["possible"], "(not a real OEIS id)")
    print("tags:", ["graph theory"])
    print()
    print("Toy predicate (NOT problem #426's actual mathematics):")
    print("  n in [0,7] with popcount(n)==2 and n even")
    print("  classical brute-force marked set:", marked)
    print()
    print(f"Grover circuit: {N_QUBITS} qubits, {iterations} iteration(s), {shots} shots")
    print("Measured counts (as integers):", dict(sorted(int_counts.items())))
    print(f"Most frequent measured outcome: {top_n}")
    print(f"Probability mass on marked set: {measured_marked_prob:.3f}")
    print()

    circuit_pass = quantum_ok
    print("CIRCUIT MECHANISM CHECK:", "PASS" if circuit_pass else "FAIL")

    print()
    print("HONEST SUMMARY:")
    print("  ran_ok = True (script executed end-to-end without error)")
    print("  verified_against_classical (of Erdos problem #426 itself) = False")
    print("    -- no real OEIS id is attached to problem #426 in the source data,")
    print("       so no genuine problem-426 sequence property was tested; the")
    print("       Grover circuit above only verifies a placeholder toy predicate")
    print("       against its own classical brute-force answer.")

    if not circuit_pass:
        print("PASS/FAIL: FAIL (toy circuit mechanism did not match classical answer)")
        sys.exit(1)
    else:
        print("PASS/FAIL: PASS (toy circuit mechanism matches classical answer;")
        print("                  problem #426 itself remains untested -- no OEIS id)")


if __name__ == "__main__":
    main()
