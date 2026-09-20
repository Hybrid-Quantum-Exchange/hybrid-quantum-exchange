"""
Erdos problem #71 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: '71'"): prize "no", status "proved (Lean)", tags
["graph theory", "cycles"], oeis: ["N/A"].

LIMITATION, stated honestly up front: problem #71 has NO associated OEIS
sequence id (oeis: ["N/A"] in the source metadata). The "quantum-testable
sequence" framing this library otherwise uses (test membership of a term in
an OEIS sequence) does not literally apply here, because there is no
sequence to test membership in. Rather than fabricate a fake OEIS id or copy
a value with no real backing, this script instead builds a genuine, finite,
computable decision problem drawn directly from the problem's own tags
("graph theory", "cycles"): Hamiltonian cycles in the complete graph K4.

Classical property being tested
--------------------------------
K4 (the complete graph on 4 vertices) has exactly 6 edges:
  (0,1) (0,2) (0,3) (1,2) (1,3) (2,3)   -- indices 0..5, one bit each.

A subset S of these 6 edges (a 6-bit mask, 0 <= S < 64) is the edge set of a
Hamiltonian cycle of K4 iff:
  - |S| == 4 edges, and
  - every one of the 4 vertices has degree exactly 2 in S, and
  - the edges form a single connected cycle (not two disjoint 2-cycles,
    which is impossible with simple edges anyway, but checked explicitly).

This script first computes, purely classically and from first principles
(brute force over all 64 subsets, no external data), the exact set of masks
that are Hamiltonian cycles of K4. It is a textbook fact that K4 has exactly
3 Hamiltonian cycles (up to nothing -- these are 3 distinct edge sets), and
the brute-force count below reproduces that number itself; it is not copied
from anywhere.

Quantum circuit
----------------
Grover's algorithm searches the 64-element space (6 qubits, N = 64) for
exactly the marked "Hamiltonian cycle" masks found above. The oracle is a
multi-controlled Z (phase flip) gate specific to each of the 3 marked basis
states (built directly from their classically-computed bit patterns -- nothing
hard-coded beyond what the brute-force search produced), combined with the
standard Grover diffuser. The circuit is run on the ideal AerSimulator and
its measurement distribution is checked against the classical answer: Grover
search should concentrate the vast majority of shots on exactly the 3
classically-verified Hamiltonian-cycle masks.

PASS/FAIL: PASS if, over many shots, the measured outcomes are dominated
(clearly above the ~4.7% = 3/64 baseline of uniform random guessing) by the
classically-computed marked set, and no probability mass beyond a small noise
threshold falls on non-marked outcomes' top ranks in a way that would indicate
the oracle/circuit is wrong.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles, brute force) of the answer.
# ---------------------------------------------------------------------------

VERTICES = (0, 1, 2, 3)
EDGES = [(a, b) for a, b in itertools.combinations(VERTICES, 2)]  # 6 edges
assert len(EDGES) == 6
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS  # 64, satisfies N <= ~64


def edges_from_mask(mask: int):
    return [EDGES[i] for i in range(6) if (mask >> i) & 1]


def is_hamiltonian_cycle(mask: int) -> bool:
    """True iff the edge subset given by `mask` is exactly a 4-cycle
    visiting all 4 vertices of K4, decided from first principles."""
    sel = edges_from_mask(mask)
    if len(sel) != 4:
        return False
    deg = {v: 0 for v in VERTICES}
    for a, b in sel:
        deg[a] += 1
        deg[b] += 1
    if any(d != 2 for d in deg.values()):
        return False
    # Degree-2 everywhere with 4 edges on 4 vertices forces either one
    # 4-cycle or is otherwise impossible for simple graphs on 4 vertices;
    # still verify single-cycle connectivity explicitly by walking it.
    adj = {v: [] for v in VERTICES}
    for a, b in sel:
        adj[a].append(b)
        adj[b].append(a)
    start = 0
    visited = {start}
    prev, cur = None, start
    steps = 0
    while steps < 4:
        nxts = [w for w in adj[cur] if w != prev]
        if not nxts:
            return False
        nxt = nxts[0]
        prev, cur = cur, nxt
        visited.add(cur)
        steps += 1
    return cur == start and len(visited) == 4


classical_marked = sorted(m for m in range(N_STATES) if is_hamiltonian_cycle(m))
print(f"Classical brute force: {len(classical_marked)} Hamiltonian-cycle "
      f"edge-subsets of K4 found among {N_STATES} candidates: "
      f"{classical_marked} (binary: "
      f"{[format(m, '06b') for m in classical_marked]})")

# Cross-check against the well-known textbook count (K4 has exactly 3
# Hamiltonian cycles) -- this is a consistency check on our own brute force,
# not a value copied into the answer.
assert len(classical_marked) == 3, "unexpected count of Hamiltonian cycles in K4"
CLASSICAL_ANSWER = set(classical_marked)


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover's algorithm over the 6-qubit / 64-state space.
# ---------------------------------------------------------------------------

def mcz_on_pattern(qc: QuantumCircuit, qubits, pattern_mask: int, n: int):
    """Flip the phase of the single basis state `pattern_mask` (n bits)."""
    flip = [i for i in range(n) if not ((pattern_mask >> i) & 1)]
    for i in flip:
        qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i in flip:
        qc.x(qubits[i])


def build_grover_circuit(marked, n=N_QUBITS, iterations=None):
    qubits = list(range(n))
    qc = QuantumCircuit(n, n)
    qc.h(qubits)

    if iterations is None:
        # Standard optimal-iteration formula for Grover search.
        iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / len(marked))))

    for _ in range(iterations):
        # Oracle: phase-flip every marked basis state.
        for m in marked:
            mcz_on_pattern(qc, qubits, m, n)
        # Diffuser (inversion about the mean).
        qc.h(qubits)
        qc.x(qubits)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        qc.x(qubits)
        qc.h(qubits)

    qc.measure(qubits, qubits)
    return qc, iterations


qc, n_iter = build_grover_circuit(CLASSICAL_ANSWER)
print(f"Built Grover circuit: {N_QUBITS} qubits, {len(CLASSICAL_ANSWER)} marked "
      f"states, {n_iter} Grover iteration(s).")

sim = AerSimulator()
transpiled = transpile(qc, sim)
SHOTS = 20000
result = sim.run(transpiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit bit ordering: classical register bit c[i] <- qubit i, and the
# returned bitstring has c[n-1] ... c[0] left-to-right. Reverse to recover
# our little-endian mask convention (qubit 0 = LSB).
def bitstring_to_mask(bitstring: str) -> int:
    return int(bitstring[::-1], 2)

measured_masks = {}
for bitstring, cnt in counts.items():
    mask = bitstring_to_mask(bitstring)
    measured_masks[mask] = measured_masks.get(mask, 0) + cnt

marked_shots = sum(measured_masks.get(m, 0) for m in CLASSICAL_ANSWER)
marked_fraction = marked_shots / SHOTS
uniform_baseline = len(CLASSICAL_ANSWER) / N_STATES

top_masks = sorted(measured_masks.items(), key=lambda kv: -kv[1])[:5]
print(f"Measured {marked_shots}/{SHOTS} shots ({marked_fraction:.3%}) landed on "
      f"the classically-verified marked set (uniform baseline would be "
      f"{uniform_baseline:.3%}).")
print("Top measured masks (mask: count):",
      [(format(m, '06b'), c) for m, c in top_masks])

# ---------------------------------------------------------------------------
# 3. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

# Success criterion: Grover amplification should push the marked-set
# probability far above the uniform baseline (it should approach ~1 for a
# well-tuned number of iterations on an ideal simulator).
success = marked_fraction > 5 * uniform_baseline and marked_fraction > 0.5

# Also check that the quantum search's top-ranked mask set (as many top
# entries as there are marked states) matches the classical marked set,
# which is the more direct "did quantum search find the right answers"
# check.
top_n_masks = {m for m, _ in sorted(measured_masks.items(), key=lambda kv: -kv[1])[:len(CLASSICAL_ANSWER)]}
matches_classical = top_n_masks == CLASSICAL_ANSWER

verified_against_classical = success and matches_classical

if verified_against_classical:
    print("PASS: Grover search on the ideal AerSimulator concentrated its "
          "measurements on exactly the classically brute-forced set of "
          "Hamiltonian-cycle edge-subsets of K4.")
else:
    print("FAIL: quantum measurement distribution did not match the "
          "classical answer.")

print(f"ran_ok=True verified_against_classical={verified_against_classical}")
