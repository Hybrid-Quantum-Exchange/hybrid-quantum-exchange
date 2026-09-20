"""
Erdos problem #759 -- quantum-testable lane.

Source metadata (data/problems.yaml, entry "number: '759'"):
    prize: no
    status: solved (2025-08-31)
    oeis: ["possible"]
    tags: ["graph theory", "chromatic number"]

LIMITATION (reported honestly, per instructions): the yaml entry for problem
759 does not carry a real OEIS sequence id -- the single tag value is the
literal placeholder string "possible", not an identifier like "A000040".
There is therefore no genuine OEIS sequence to test membership/terms of for
this problem, and no classical "known term" to check a quantum result
against that would actually come from problem 759's own data.

Rather than fabricate an OEIS id or copy a value with no real connection to
this problem, this script instead builds the best honest quantum instance
supported by problem 759's *tags* alone ("graph theory", "chromatic
number"): it tests, with a real Grover search circuit on the ideal
AerSimulator, whether a small fixed graph is 3-colorable -- a canonical
finite/computable chromatic-number question. The classical answer is
computed here from first principles by brute-force enumeration over all
3-colorings of the graph (not copied from any table), and the quantum
result is compared against it.

Graph used (5-cycle C5, a standard example with chromatic number 3):
    vertices: 0,1,2,3,4
    edges: (0,1) (1,2) (2,3) (3,4) (4,0)

Property tested: "C5 has a proper 3-coloring" (true, since chi(C5) = 3).
Encoding: each vertex gets 2 qubits (2-bit color in {0,1,2}, code 3 unused
and excluded by the oracle), for 10 qubits total. Grover's algorithm
amplifies the basis states encoding valid proper 3-colorings; we then
measure and check that a valid coloring (per the classical edge-conflict
definition) is found with high probability, matching the classical brute
force existence answer (True).

Because this specific instance is derived from the tags only (no OEIS id
exists for problem 759), verified_against_classical here means: the
quantum search's measured outcome, decoded and checked against the same
classical proper-coloring predicate, agrees with the independently computed
brute-force classical answer -- not that it reproduces a term of a named
OEIS sequence for this problem.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# Problem instance: C5, and the classical (first-principles) ground truth.
# ---------------------------------------------------------------------------

N_VERTICES = 5
EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
COLORS = (0, 1, 2)  # 3-coloring; code "3" (bits 11) is invalid/unused


def is_proper_coloring(coloring):
    """coloring: tuple of length N_VERTICES, each entry in {0,1,2,3}."""
    if any(c == 3 for c in coloring):
        return False
    return all(coloring[u] != coloring[v] for (u, v) in EDGES)


def classical_brute_force():
    """Enumerate every 3-coloring of C5 and return the valid ones."""
    valid = []
    for coloring in itertools.product(COLORS, repeat=N_VERTICES):
        if is_proper_coloring(coloring):
            valid.append(coloring)
    return valid


VALID_COLORINGS = classical_brute_force()
CLASSICAL_EXISTS = len(VALID_COLORINGS) > 0
NUM_VALID = len(VALID_COLORINGS)
SEARCH_SPACE_SIZE = 4 ** N_VERTICES  # each vertex has a 2-qubit register

print(f"Classical brute force: {NUM_VALID} valid proper 3-colorings out of "
      f"{SEARCH_SPACE_SIZE} possible 2-bit-per-vertex assignments.")
print(f"Classical answer: C5 is 3-colorable = {CLASSICAL_EXISTS}")

# ---------------------------------------------------------------------------
# Quantum circuit: Grover search over the 10-qubit (5 vertices x 2 bits)
# assignment space, with an oracle that marks proper 3-colorings of C5.
# ---------------------------------------------------------------------------

QUBITS_PER_VERTEX = 2
N_QUBITS = N_VERTICES * QUBITS_PER_VERTEX  # 10


def vertex_qubits(v):
    return [v * QUBITS_PER_VERTEX, v * QUBITS_PER_VERTEX + 1]


def build_oracle():
    """Phase-flip oracle marking basis states that encode a proper coloring.

    A state is marked (bad -> excluded from "valid") if:
      - any vertex uses code 3 (both qubits = 1), OR
      - any edge has both endpoints with equal color code.
    We build a circuit that flags "conflict" into ancillas and uses a
    multi-controlled Z conditioned on "no conflict at all" via De Morgan:
    we instead directly flag "is invalid" ancillas and phase-flip when
    ALL invalid-flags are 0, using X-sandwiching (flip flags, MCZ on all-0,
    flip back).
    """
    n_code3_flags = N_VERTICES
    qc = QuantumCircuit(N_QUBITS + n_code3_flags + len(EDGES), name="oracle")
    # ancilla layout: [0..N_VERTICES-1] = one "uses invalid code 3" flag per
    # vertex (kept separate per vertex so two such vertices can never cancel
    # each other out), [N_VERTICES..] = one flag per edge (1 if that edge
    # conflicts).
    code3_flags = [N_QUBITS + v for v in range(N_VERTICES)]
    edge_flags = [N_QUBITS + n_code3_flags + i for i in range(len(EDGES))]

    # Flag each vertex using invalid code "3" (both its qubits = 1)
    for v in range(N_VERTICES):
        q0, q1 = vertex_qubits(v)
        qc.ccx(q0, q1, code3_flags[v])

    # Flag equal-color conflicts per edge: bit0 equal AND bit1 equal
    for i, (u, v) in enumerate(EDGES):
        uq0, uq1 = vertex_qubits(u)
        vq0, vq1 = vertex_qubits(v)
        # temp: XOR each bit pair into vq (reversible), then check both 0
        qc.cx(uq0, vq0)
        qc.cx(uq1, vq1)
        # edge_flags[i] = 1 iff vq0==0 and vq1==0 (i.e. colors equal)
        qc.x(vq0)
        qc.x(vq1)
        qc.ccx(vq0, vq1, edge_flags[i])
        qc.x(vq0)
        qc.x(vq1)
        # uncompute the XOR
        qc.cx(uq1, vq1)
        qc.cx(uq0, vq0)

    return qc, code3_flags, edge_flags


def build_full_oracle_with_phase():
    """Build the oracle as a phase-flip via a dedicated phase-kickback
    ancilla prepared in |-> by the caller."""
    oracle, code3_flags, edge_flags = build_oracle()
    all_flags = code3_flags + edge_flags
    n_ancilla = len(code3_flags) + len(edge_flags)
    total_qubits = N_QUBITS + n_ancilla + 1  # +1 phase-kick ancilla (|-> state)
    phase_q = N_QUBITS + n_ancilla

    qc = QuantumCircuit(total_qubits, name="oracle_phase")
    qc.compose(oracle, range(N_QUBITS + n_ancilla), inplace=True)
    qc.x(all_flags)
    qc.mcx(all_flags, phase_q)
    qc.x(all_flags)
    qc.compose(oracle.inverse(), range(N_QUBITS + n_ancilla), inplace=True)
    return qc, phase_q, n_ancilla


def full_grover_circuit(iterations):
    oracle_qc, phase_q, n_ancilla = build_full_oracle_with_phase()
    total_qubits = N_QUBITS + n_ancilla + 1

    qc = QuantumCircuit(total_qubits, N_QUBITS)
    qc.h(range(N_QUBITS))
    qc.x(phase_q)
    qc.h(phase_q)

    for _ in range(iterations):
        qc.compose(oracle_qc, range(total_qubits), inplace=True)
        # diffusion on search register only
        qc.h(range(N_QUBITS))
        qc.x(range(N_QUBITS))
        qc.h(N_QUBITS - 1)
        qc.mcx(list(range(N_QUBITS - 1)), N_QUBITS - 1)
        qc.h(N_QUBITS - 1)
        qc.x(range(N_QUBITS))
        qc.h(range(N_QUBITS))

    qc.h(phase_q)
    qc.x(phase_q)
    qc.measure(range(N_QUBITS), range(N_QUBITS))
    return qc


def decode(bitstring):
    """Qiskit bitstrings are little-endian in classical-register order
    (rightmost char = qubit 0). Recover the per-vertex 2-bit codes."""
    bits = bitstring[::-1]  # bits[i] = qubit i
    coloring = []
    for v in range(N_VERTICES):
        q0, q1 = vertex_qubits(v)
        code = int(bits[q0]) + 2 * int(bits[q1])
        coloring.append(code)
    return tuple(coloring)


def main():
    # Grover iteration count: M valid states out of N = 4^5 = 1024.
    N = SEARCH_SPACE_SIZE
    M = NUM_VALID
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Search space N={N}, valid solutions M={M}, Grover iterations={iterations}")

    qc = full_grover_circuit(iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Decode every observed bitstring and check it against the SAME
    # classical proper-coloring predicate used for the brute force above.
    hit_shots = 0
    total_shots = 0
    best_bitstring, best_count = max(counts.items(), key=lambda kv: kv[1])
    for bitstring, count in counts.items():
        total_shots += count
        coloring = decode(bitstring)
        if is_proper_coloring(coloring):
            hit_shots += count

    hit_fraction = hit_shots / total_shots
    best_coloring = decode(best_bitstring)
    best_is_valid = is_proper_coloring(best_coloring)

    print(f"Most frequent measured outcome: {best_bitstring} -> coloring {best_coloring}, "
          f"valid={best_is_valid}, count={best_count}/{shots}")
    print(f"Fraction of all shots decoding to a valid proper 3-coloring: {hit_fraction:.4f}")

    # Quantum result: did Grover search successfully concentrate probability
    # on valid states, and does the most likely measured outcome match a
    # genuine classical proper 3-coloring (cross-checked against the
    # independently brute-forced VALID_COLORINGS list)?
    quantum_found_valid = best_is_valid and (best_coloring in VALID_COLORINGS)
    quantum_answer_exists = hit_fraction > 0.5  # majority of shots are valid colorings

    verified = (quantum_answer_exists == CLASSICAL_EXISTS) and quantum_found_valid

    print(f"Classical existence answer : {CLASSICAL_EXISTS}")
    print(f"Quantum-derived existence  : {quantum_answer_exists} "
          f"(majority-of-shots valid, best outcome itself valid={best_is_valid})")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
