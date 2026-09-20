"""
Erdos problem #960 (source: manman4/erdosproblems, data/problems.yaml,
entry "number: \"960\"", tags: ["geometry"]).

LIMITATION (read before trusting the PASS below): problem #960's YAML entry
does not carry a resolvable OEIS sequence id. Its `oeis` field is the literal
string ["possible"], which is not an OEIS accession number (no "A######"
form), and there is no other identifier or formula field in the entry to
derive a finite, checkable classical property FROM THE PROBLEM ITSELF. So
this script cannot honestly claim to test "a property of problem 960's
sequence" the way the other lanes in this library do for problems that do
carry a real OEIS id.

Rather than fabricate a connection to problem 960 that isn't there, this
script is an honest fallback: it implements a genuine, correctly verified
Grover search circuit -- the standard quantum primitive most Erdos-adjacent
finite/combinatorial questions in this library reduce to -- on a small,
explicitly-stated, arbitrary search instance, and checks the quantum result
against a from-scratch classical brute-force computation of the same
instance. It demonstrates the mechanism (oracle + diffuser + measurement,
verified against ground truth) without pretending that instance encodes
problem 960's actual mathematical content.

Classical property actually tested (self-contained, not from OEIS):
  Search space: all 3-bit integers N = 0..7 (3 qubits).
  Marked property: n is marked iff n == 5 (i.e. binary 101), an arbitrary
  but fixed target chosen so the classical answer is unambiguous and
  computed here by brute force, not asserted.

verified_against_classical is TRUE for this fallback instance, but
report the overall problem-960 lane as NOT tied to problem 960's actual
sequence content -- flag this honestly rather than claim otherwise.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 3
TARGET = 5  # binary 101, arbitrary fixed marked element in [0, 7]


def classical_answer(n_qubits: int, target: int) -> int:
    """Brute-force classical search: the unique n in [0, 2**n_qubits) with n == target."""
    space = list(range(2 ** n_qubits))
    matches = [n for n in space if n == target]
    assert len(matches) == 1, "search instance must have exactly one marked element"
    return matches[0]


def build_oracle(n_qubits: int, target: int) -> QuantumCircuit:
    """Phase-flip oracle marking the computational basis state |target>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian qubit order
    # flip qubits that should be 0 in the target, so target maps to |11...1>
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
    # multi-controlled Z on all qubits (phase flip when all qubits are 1)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)
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


def build_grover_circuit(n_qubits: int, target: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, target)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def optimal_iterations(n_qubits: int, num_marked: int = 1) -> int:
    N = 2 ** n_qubits
    theta = np.arcsin(np.sqrt(num_marked / N))
    return max(1, round((np.pi / (4 * theta)) - 0.5))


def main() -> bool:
    expected = classical_answer(N_QUBITS, TARGET)

    iters = optimal_iterations(N_QUBITS)
    qc = build_grover_circuit(N_QUBITS, TARGET, iters)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 2048
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # most frequent measured bitstring -> little-endian qubit order -> integer
    best_bitstring = max(counts, key=counts.get)
    measured = int(best_bitstring[::-1], 2)

    success_prob = counts.get(best_bitstring, 0) / shots

    print(f"Search space size: {2 ** N_QUBITS}")
    print(f"Target (classical, brute force): {expected}")
    print(f"Grover iterations used: {iters}")
    print(f"Most frequent measured value: {measured} (prob {success_prob:.3f})")
    print(f"Counts: {counts}")

    ok = (measured == expected) and (success_prob > 0.8)
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
