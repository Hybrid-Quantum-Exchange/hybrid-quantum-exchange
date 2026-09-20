"""
Erdos problem #583 -- Qiskit quantum-testable lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: '583'"):
    tags: ["graph theory"]
    oeis: ["N/A"]
    informal_status: falsifiable (as of 2025-08-31)

HONEST LIMITATION: problem #583 carries no OEIS sequence id ("N/A"). The
broader task family this lane belongs to asks for a small, finite, computable
property of an OEIS sequence tied to the problem, verified by a real quantum
circuit. With no OEIS sequence attached to #583, that specific request cannot
be fulfilled as stated for this problem number.

Best-effort substitute, honestly labelled as such: the only structured
content #583 carries is its tag, "graph theory". Rather than fabricate an
OEIS value, this script builds a genuine, small, finite, classically
verifiable graph-theory decision problem -- in the spirit of the extremal /
combinatorial questions Erdos problems are typically about -- and solves it
with a real Grover search circuit on the ideal AerSimulator, cross-checked
against a brute-force classical computation done from first principles in
this same script.

Chosen instance: label the 6 possible edges of K4 (vertices 1..4) in fixed
order EDGES = [12,13,14,23,24,34]. A labeled subgraph of K4 is encoded as a
6-bit string, bit i on iff edge EDGES[i] is present. There are N = 2^6 = 64
such subgraphs.

Property tested: "this subgraph contains the specific triangle {12,13,23}",
i.e. bits for edges 12, 13, and 23 are all 1 (the other three edges are
free). This is a small, finite, exactly-computable search problem well
suited to Grover's algorithm: M = 2^3 = 8 of the N = 64 subgraphs satisfy it.

The script:
  1. Computes the exact classical answer by brute force over all 64
     edge-subsets (first principles, no lookup table).
  2. Builds a genuine 3-input reversible AND oracle (a Toffoli-of-2 into an
     ancilla, then a Toffoli combining that ancilla with the third edge bit,
     phase-kicked onto an ancilla prepared in the |-> state, then uncomputed)
     -- a real boolean circuit, not a hard-coded answer.
  3. Runs the standard Grover diffusion operator, oracle+diffusion repeated
     for the classically-computed optimal number of iterations.
  4. Measures 4096 shots and checks the quantum result against the classical
     answer two ways: (a) the most frequent measured edge-subset is genuinely
     one of the 8 classical solutions, and (b) the aggregate probability mass
     landing on the 8 true solutions is close to the value Grover's own
     theory predicts for M=8, N=64, well above a strict acceptance bound.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force).
# ---------------------------------------------------------------------------

EDGES = ["12", "13", "14", "23", "24", "34"]  # bit i <-> EDGES[i], i = 0..5
EDGE_INDEX = {e: i for i, e in enumerate(EDGES)}
N_QUBITS = len(EDGES)  # 6
N = 2 ** N_QUBITS       # 64

TARGET_TRIANGLE = ("12", "13", "23")
A_BIT, B_BIT, C_BIT = (EDGE_INDEX[e] for e in TARGET_TRIANGLE)


def int_to_bits(x, n=N_QUBITS):
    """Little-endian: bit i = (x >> i) & 1, matching Qiskit's qubit-0-first
    bitstring convention used later when reading measurement results."""
    return tuple((x >> i) & 1 for i in range(n))


def has_target_triangle(bits):
    return bits[A_BIT] == 1 and bits[B_BIT] == 1 and bits[C_BIT] == 1


classical_solutions = {x for x in range(N) if has_target_triangle(int_to_bits(x))}
M = len(classical_solutions)
assert M == 8, f"sanity check failed: expected 8 solutions, got {M}"

theta = math.asin(math.sqrt(M / N))
optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
predicted_success_prob = math.sin((2 * optimal_iters + 1) * theta) ** 2

print(f"[classical] N = {N} labeled 6-edge subgraphs of K4")
print(f"[classical] M = {M} subgraphs contain the triangle {TARGET_TRIANGLE}")
print(f"[classical] Grover optimal iterations for this M,N: {optimal_iters}")
print(f"[classical] Grover-predicted success probability: {predicted_success_prob:.4f}")

# ---------------------------------------------------------------------------
# 2. Quantum circuit: genuine reversible oracle + Grover diffusion.
# ---------------------------------------------------------------------------

edge_reg = QuantumRegister(N_QUBITS, "edge")
scratch = AncillaRegister(1, "scratch")   # holds (edge_a AND edge_b)
phase_anc = AncillaRegister(1, "phase")   # phase-kickback ancilla, held in |->

qc = QuantumCircuit(edge_reg, scratch, phase_anc, name="grover_triangle_in_k4")

# Uniform superposition over all 64 edge-subsets.
qc.h(edge_reg)

# Phase-kickback ancilla in the |-> eigenstate of X, so a controlled-X on it
# acts as a controlled phase flip (the standard Grover phase-oracle trick).
qc.x(phase_anc[0])
qc.h(phase_anc[0])


def apply_triangle_oracle(circuit):
    # scratch = edge_a AND edge_b
    circuit.ccx(edge_reg[A_BIT], edge_reg[B_BIT], scratch[0])
    # phase_anc ^= (scratch AND edge_c) = (a AND b AND c); since phase_anc is
    # in |->, this toggling applies a genuine -1 phase exactly on the 8
    # states where edges A_BIT, B_BIT, C_BIT are all 1.
    circuit.ccx(scratch[0], edge_reg[C_BIT], phase_anc[0])
    # Uncompute scratch so it is |0> again for the next Grover iteration.
    circuit.ccx(edge_reg[A_BIT], edge_reg[B_BIT], scratch[0])


def apply_diffusion(circuit, reg):
    circuit.h(reg)
    circuit.x(reg)
    circuit.h(reg[-1])
    circuit.mcx(list(reg[:-1]), reg[-1])
    circuit.h(reg[-1])
    circuit.x(reg)
    circuit.h(reg)


for _ in range(optimal_iters):
    apply_triangle_oracle(qc)
    apply_diffusion(qc, edge_reg)

# Restore phase ancilla (not strictly required before measurement, but keeps
# the circuit fully reversible/clean) and measure only the edge register.
qc.h(phase_anc[0])
qc.x(phase_anc[0])

qc.measure_all()

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

SHOTS = 4096
backend = AerSimulator()
job = backend.run(qc, shots=SHOTS)
result = job.result()
counts = result.get_counts()

# Qiskit's measure_all bitstrings are "c[n-1] ... c[1] c[0]" over ALL
# classical bits in creation order (edge[0..5], scratch, phase); the
# rightmost 6 characters are the edge register (edge[0] is the rightmost of
# those 6), matching int_to_bits's little-endian convention above.
def edge_bits_from_outcome(bitstring):
    edge_part = bitstring[-N_QUBITS:]          # 6 rightmost chars: scratch/phase excluded
    # edge_part[-1] is edge[0], edge_part[-2] is edge[1], etc.
    return tuple(int(ch) for ch in reversed(edge_part))


def edge_bits_to_int(bits):
    return sum(b << i for i, b in enumerate(bits))


outcome_counts = {}
for bitstring, c in counts.items():
    bits = edge_bits_from_outcome(bitstring)
    x = edge_bits_to_int(bits)
    outcome_counts[x] = outcome_counts.get(x, 0) + c

most_common_x = max(outcome_counts, key=outcome_counts.get)
most_common_bits = int_to_bits(most_common_x)

success_shots = sum(c for x, c in outcome_counts.items() if x in classical_solutions)
measured_success_prob = success_shots / SHOTS

print(f"[quantum] most frequent measured subgraph (int): {most_common_x}")
print(f"[quantum] most frequent measured edges present: "
      f"{[EDGES[i] for i, b in enumerate(most_common_bits) if b]}")
print(f"[quantum] measured success probability (mass on true solutions): "
      f"{measured_success_prob:.4f}")

# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer and report PASS/FAIL.
# ---------------------------------------------------------------------------

top_result_correct = most_common_x in classical_solutions
prob_close_to_prediction = abs(measured_success_prob - predicted_success_prob) < 0.15
prob_well_above_uniform = measured_success_prob > 3 * (M / N)  # uniform baseline is M/N

verified = top_result_correct and prob_close_to_prediction and prob_well_above_uniform

print()
print(f"top_result_correct:        {top_result_correct}")
print(f"prob_close_to_prediction:  {prob_close_to_prediction} "
      f"(measured={measured_success_prob:.4f}, predicted={predicted_success_prob:.4f})")
print(f"prob_well_above_uniform:   {prob_well_above_uniform} "
      f"(measured={measured_success_prob:.4f}, uniform baseline={M/N:.4f})")
print()
print("PASS" if verified else "FAIL")
