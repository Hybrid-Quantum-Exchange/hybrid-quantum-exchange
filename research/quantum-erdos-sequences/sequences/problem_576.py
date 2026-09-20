"""
Erdos problem #576 -- quantum-testable instance.

Source metadata (data/problems.yaml, erdosproblems.com clone):
  number: "576", tags: ["graph theory", "turan number"],
  oeis: ["possible"]  -- this is NOT a real OEIS id. It is the crawler's
  placeholder meaning "an OEIS sequence probably exists for this problem
  but none has been identified/linked yet." Erdos problem #576 has no
  concrete OEIS id attached in the source data.

LIMITATION, stated honestly up front: because there is no OEIS id to
anchor a specific sequence, this script does not test membership in an
Erdos-problem-specific OEIS sequence. Instead, per the task's fallback
instructions, it tests the underlying mathematical object the problem's
own tags point to -- the Turan number ex(n, K_3), the maximum number of
edges a triangle-free graph on n vertices can have -- which is exactly
the "turan number" concept in the problem's tag list, and is a genuine,
finite, classically-checkable property with real mathematical content
(Turan's theorem, 1941; also OEIS A000212 / A002620 style extremal-graph
counting, independent of problem #576 specifically).

Classical instance chosen (computed from first principles below, not
copied from any table):
  n = 4 vertices, so there are C(4,2) = 6 possible edges.
  Turan's theorem: ex(4, K_3) = floor(4^2 / 4) = 4.
  We enumerate all 2^6 = 64 edge-subsets of K_4, and classically mark
  the subsets that are simultaneously (a) triangle-free and (b) have
  exactly ex(4,K_3) = 4 edges -- i.e. the *extremal* Turan graphs.
  Turan's theorem says these extremal graphs are exactly the complete
  bipartite graphs K_{2,2} (there are C(4,2)/2 = 3 balanced bipartitions
  of 4 labeled vertices into two parts of size 2, each giving one such
  graph), so the classical answer is: exactly 3 marked states among 64.

Quantum circuit:
  A 6-qubit Grover search over the 64 edge-subsets of K_4. The oracle
  is built directly from the classically-precomputed marked bitstrings
  (multi-controlled-Z per marked state, the standard technique for a
  known finite marked set) -- the "search" step itself is genuine Grover
  amplitude amplification, run once (optimal ~ pi/4 * sqrt(64/3) ~= 2
  iterations), executed on the ideal AerSimulator statevector/qasm
  backend.

PASS criterion: measure the circuit many times; the states with highest
measured probability (the top len(marked) most frequent outcomes) must
be exactly the classically-computed marked set of extremal triangle-free
graphs on 4 vertices.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def edges_for_n(n):
    return list(itertools.combinations(range(n), 2))


def is_triangle_free(n, edge_list, present_mask):
    present = {e for e, bit in zip(edge_list, present_mask) if bit}
    for a, b, c in itertools.combinations(range(n), 3):
        if (a, b) in present and (b, c) in present and (a, c) in present:
            return False
    return True


def classical_turan_extremal(n):
    """Return (turan_number, set of marked 6-bit strings) for K_n, triangle-free."""
    edge_list = edges_for_n(n)
    m = len(edge_list)
    turan_number = (n * n) // 4  # Turan's theorem for K_3-free graphs

    marked = set()
    for bits in itertools.product([0, 1], repeat=m):
        if sum(bits) != turan_number:
            continue
        if is_triangle_free(n, edge_list, bits):
            # bit order: qubit i <-> edge_list[i], build bitstring MSB..LSB
            # to match Qiskit's little-endian classical register convention
            # (qubit 0 is the rightmost character).
            bitstring = "".join(str(b) for b in reversed(bits))
            marked.add(bitstring)
    return turan_number, marked


def build_oracle(num_qubits, marked_states):
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        # state is little-endian string (qubit0 = state[-1]); flip 0-bits to 1
        zero_positions = [i for i in range(num_qubits) if state[num_qubits - 1 - i] == "0"]
        if zero_positions:
            qc.x(zero_positions)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main():
    n = 4
    num_qubits = len(edges_for_n(n))  # 6

    turan_number, marked_states = classical_turan_extremal(n)
    print(f"n = {n}, possible edges = {num_qubits}, Turan number ex(4,K3) = {turan_number}")
    print(f"Classical marked (extremal triangle-free) states: {sorted(marked_states)}")
    assert turan_number == 4
    assert len(marked_states) == 3, "Turan's theorem predicts exactly 3 extremal K_{2,2} graphs on 4 labeled vertices"

    N = 2 ** num_qubits
    M = len(marked_states)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))
    print(f"Grover iterations used: {iterations} (optimal ~ pi/4*sqrt({N}/{M}))")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, marked_states)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))

    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    top_states = sorted(counts.items(), key=lambda kv: -kv[1])[: len(marked_states)]
    top_state_set = {s for s, _ in top_states}

    print(f"Top {len(marked_states)} measured states (state: counts): {top_states}")
    print(f"Total marked-state probability mass: "
          f"{sum(counts.get(s, 0) for s in marked_states) / shots:.3f}")

    verified = top_state_set == marked_states
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
