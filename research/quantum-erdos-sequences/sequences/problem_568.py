"""
Erdos problem #568 -- quantum-testable sequence entry.

Source: /home/user/manman4/erdosproblems/data/problems.yaml, block
"- number: \"568\"":
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["graph theory", "ramsey theory"]

LIMITATION: problem #568 carries no OEIS id (oeis: ["N/A"]), so there is no
literal sequence term to reproduce from OEIS. This script instead builds a
genuine, independently-derived finite/computable property drawn directly from
the problem's own tags (graph theory, Ramsey theory), and verifies it with a
real Grover search circuit rather than fabricating an OEIS value.

Classical property being tested
--------------------------------
Fact underlying the Ramsey number R(3,3) = 6: K5 (the complete graph on 5
vertices) CAN be 2-edge-colored with no monochromatic triangle, while K6
cannot. K5 has C(5,2) = 10 edges and C(5,3) = 10 triangles. Each of the 2^10
= 1024 edge-colorings is represented as a 10-bit string (bit i = color of
edge i, in a fixed enumeration of the edges). A coloring is "good" if none
of the 10 triangles is monochromatic.

This script:
  1. Enumerates all 1024 edge-colorings of K5 classically (first principles,
     no library/OEIS lookup) and computes the exact set of "good" (no
     monochromatic-triangle) colorings. This count is the classical answer.
  2. Builds a Grover search circuit over 10 qubits whose oracle marks exactly
     those good colorings (via a per-state multi-controlled phase flip built
     from the classically-enumerated set), with the standard 10-qubit
     diffuser, iterated the near-optimal number of times.
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured bitstrings, and checks they are members of the classically
     computed good set.
  4. Prints PASS if Grover search boosted good colorings to be the dominant
     measurement outcomes (i.e. quantum search finds real members of the
     classical solution set), else FAIL.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_VERTICES = 5
EDGES = list(itertools.combinations(range(N_VERTICES), 2))  # 10 edges
TRIANGLES = list(itertools.combinations(range(N_VERTICES), 3))  # 10 triangles
N_QUBITS = len(EDGES)  # 10


def edge_index(u, v):
    return EDGES.index((u, v) if u < v else (v, u))


def is_good_coloring(bits):
    """bits: int in [0, 2**N_QUBITS). bit i (LSB-first) = color of EDGES[i]."""
    for (a, b, c) in TRIANGLES:
        e1 = edge_index(a, b)
        e2 = edge_index(b, c)
        e3 = edge_index(a, c)
        c1 = (bits >> e1) & 1
        c2 = (bits >> e2) & 1
        c3 = (bits >> e3) & 1
        if c1 == c2 == c3:
            return False
    return True


def classical_good_set():
    good = []
    for bits in range(2 ** N_QUBITS):
        if is_good_coloring(bits):
            good.append(bits)
    return good


def bits_to_bitstring(bits, n=N_QUBITS):
    # Qiskit's measured bitstrings print qubit (n-1) leftmost, qubit 0
    # rightmost -- i.e. plain big-endian formatting of the integer whose
    # bit i (from the LSB) is qubit i's value, which is exactly `bits` here.
    return format(bits, f"0{n}b")


def add_phase_oracle_mark(qc, target_bits, n=N_QUBITS):
    """Flip the phase of the single computational basis state target_bits."""
    zero_positions = [i for i in range(n) if ((target_bits >> i) & 1) == 0]
    for i in zero_positions:
        qc.x(i)
    # Multi-controlled Z across all n qubits: use qubit n-1 as phase target.
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle(good_states, n=N_QUBITS):
    qc = QuantumCircuit(n, name="oracle")
    for state in good_states:
        add_phase_oracle_mark(qc, state, n)
    return qc


def build_diffuser(n=N_QUBITS):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def main():
    good_states = classical_good_set()
    n_good = len(good_states)
    n_total = 2 ** N_QUBITS
    print(f"Classical answer: {n_good} good (no monochromatic triangle) "
          f"2-edge-colorings of K5 out of {n_total} total.")
    assert n_good > 0, "K5 must admit a triangle-free-monochromatic coloring (R(3,3)=6 fact)"

    # Near-optimal number of Grover iterations for this search-space/marked ratio.
    theta = math.asin(math.sqrt(n_good / n_total))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Running Grover search with {iterations} iteration(s) over {N_QUBITS} qubits.")

    oracle = build_oracle(good_states)
    diffuser = build_diffuser()

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096).result()
    counts = result.get_counts()

    # Sort measured bitstrings by frequency, most common first.
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = min(n_good, 8)
    top_states = sorted_counts[:top_k]

    good_bitstrings = {bits_to_bitstring(b) for b in good_states}

    print("Top measured outcomes (bitstring: count):")
    for bs, cnt in top_states:
        print(f"  {bs}: {cnt}  {'(good)' if bs in good_bitstrings else '(NOT good)'}")

    top_mass = sum(cnt for bs, cnt in sorted_counts if bs in good_bitstrings)
    total_shots = sum(counts.values())
    good_fraction = top_mass / total_shots
    print(f"Fraction of shots landing on a classically-verified good coloring: "
          f"{good_fraction:.4f}")

    all_top_good = all(bs in good_bitstrings for bs, _ in top_states)
    boosted = good_fraction > (n_good / n_total) * 3  # meaningfully amplified vs uniform baseline

    verified = all_top_good and boosted

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
