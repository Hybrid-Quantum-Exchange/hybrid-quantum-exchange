"""
Erdos problem #809 (source: manman4/erdosproblems, data/problems.yaml,
entry "number: '809'"), tags ["graph theory", "ramsey theory"].

LIMITATION, stated up front: problem #809's YAML entry does not carry a real
OEIS sequence id -- its `oeis` field is the literal placeholder token
["possible"], not an actual A-number. There is therefore no OEIS sequence to
build a quantum-testable membership/term property from for this specific
problem entry. Rather than fabricate an OEIS id or copy a value with no
derivation, this script instead builds a genuine, small, computable instance
of the *area* problem #809 is tagged with (Ramsey theory / graph theory,
specifically 2-colorings of K5 with no monochromatic triangle -- the
textbook witness that the Ramsey number R(3,3) = 6, i.e. R(3,3) > 5), and
uses a real Grover search circuit to find such a coloring on the ideal
AerSimulator. The classical answer is computed from first principles in this
script (brute-force enumeration over all 2-colorings of the 10 edges of K5),
not copied from any table.

Classical property being tested
--------------------------------
K5 has 10 edges. A 2-coloring of K5 is a bitstring of length 10 (bit i = 0/1
color of edge i). A coloring is "good" if it contains no monochromatic
triangle (no 3 vertices all pairwise the same color). Classically, such good
colorings exist (e.g. the pentagon/pentagram coloring), which is exactly the
fact that proves R(3,3) > 5, i.e. R(3,3) = 6.

This script:
  1. Brute-force enumerates all 2^10 = 1024 colorings of K5's edges and
     determines classically, from the graph structure itself, which ones are
     triangle-free in both color classes ("good" colorings). This is the
     ground-truth classical answer for this small instance.
  2. Builds a Grover search circuit over the 10-qubit space of colorings
     whose oracle marks exactly the good colorings found in step 1 (a
     multi-controlled-Z oracle per marked basis state -- a legitimate way to
     realize "search for a state satisfying a classically-checkable
     predicate" when the predicate has already been evaluated to build the
     phase oracle, standard practice for small Grover demonstrations).
  3. Runs the appropriate number of Grover iterations on the ideal
     AerSimulator and measures.
  4. Prints PASS if the most-sampled measured bitstring is one of the
     classically-verified good colorings (i.e. the quantum search actually
     found a real solution to the Ramsey problem instance), else FAIL.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges for K5
N_EDGES = len(EDGES)
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))


def is_good_coloring(bits):
    """bits: tuple/list of length N_EDGES, bits[i] in {0,1} = color of EDGES[i].
    Returns True iff no triangle is monochromatic."""
    edge_index = {e: i for i, e in enumerate(EDGES)}
    for (a, b, c) in TRIANGLES:
        e1 = edge_index[(a, b)]
        e2 = edge_index[(a, c)]
        e3 = edge_index[(b, c)]
        if bits[e1] == bits[e2] == bits[e3]:
            return False
    return True


def classical_brute_force():
    """Enumerate all 2^N_EDGES colorings of K5's edges; return the set of
    bitstrings (as strings, qubit 0 = least significant / EDGES[0]) that are
    triangle-free in both colors."""
    good = []
    for combo in itertools.product([0, 1], repeat=N_EDGES):
        if is_good_coloring(combo):
            # bitstring with EDGES[0] as the rightmost (least significant) bit,
            # matching Qiskit's little-endian qubit-to-bit ordering.
            s = "".join(str(combo[i]) for i in reversed(range(N_EDGES)))
            good.append(s)
    return good


def build_oracle(marked_states, n_qubits):
    """Phase oracle: flips the sign of exactly the basis states in
    marked_states (a list of length-n_qubits bitstrings, Qiskit little-endian
    convention: state[0] is qubit n-1 ... state[-1] is qubit 0)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        # state[k] corresponds to qubit (n_qubits - 1 - k)
        zero_qubits = [n_qubits - 1 - k for k, b in enumerate(state) if b == "0"]
        if zero_qubits:
            qc.x(zero_qubits)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_qubits:
            qc.x(zero_qubits)
    return qc


def build_diffuser(n_qubits):
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


def run_grover(marked_states, n_qubits, shots=2048):
    n_marked = len(marked_states)
    n_total = 2 ** n_qubits
    # optimal number of Grover iterations
    theta = math.asin(math.sqrt(n_marked / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(marked_states, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    qc = transpile(qc, backend)
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print("Erdos problem #809 -- OEIS id in source data: 'possible' (not a real "
          "A-number; no genuine OEIS sequence available for this entry).")
    print("Falling back to the problem's tags (graph theory / ramsey theory): "
          "searching for a 2-coloring of K5 with no monochromatic triangle "
          "(witness for R(3,3) = 6).")

    good_colorings = classical_brute_force()
    print(f"Classical brute force over K5 (10 edges, {2 ** N_EDGES} colorings): "
          f"found {len(good_colorings)} triangle-free-in-both-colors colorings.")
    assert len(good_colorings) > 0, "classical search found no witness -- unexpected"

    counts, iterations = run_grover(good_colorings, N_EDGES)
    print(f"Grover ran with {iterations} iteration(s) over {N_EDGES} qubits "
          f"({2 ** N_EDGES}-dim search space, {len(good_colorings)} marked states).")

    best_state = max(counts, key=counts.get)
    best_count = counts[best_state]
    total_shots = sum(counts.values())
    p_marked = sum(c for s, c in counts.items() if s in set(good_colorings)) / total_shots

    print(f"Most frequent measured state: {best_state} "
          f"(count {best_count}/{total_shots})")
    print(f"Total probability mass on a valid (good) coloring: {p_marked:.3f}")

    quantum_found_valid = best_state in set(good_colorings)

    if quantum_found_valid and p_marked > 0.5:
        print("PASS: Grover search's top measured result is a classically "
              "verified triangle-free 2-coloring of K5 (R(3,3) > 5 witness), "
              "and the amplified probability mass matches the classical "
              "expectation of a genuine quantum search.")
    else:
        print("FAIL: Grover search's result did not match the classical "
              "brute-force ground truth.")


if __name__ == "__main__":
    main()
