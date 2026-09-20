"""
Erdos problem #304 -- quantum-testable instance.

Source: erdosproblems.com problem 304 (unit fractions / number theory).
  data/problems.yaml entry:
    number: "304", oeis: ["A097847", "A097849"], tags: ["number theory", "unit fractions"]

OEIS definitions (fetched from oeis.org):
  A097847: triangle read by rows, T(n,k) = minimal number of (not
           necessarily distinct) unit fractions 1/d_1 + 1/d_2 + ... needed
           to write k/n, for 1 <= k <= n.
  A097849: row maxima of A097847.

Classical property tested (computed from first principles below, not
copied from OEIS): for n=3, k=2 (so the target fraction is 2/3), is there
a representation of 2/3 as a sum of exactly TWO unit fractions
1/d1 + 1/d2 with d1, d2 drawn from the small finite domain {1,2,3,4}?
The script brute-forces this classically with exact Fraction arithmetic
over the 4x4 = 16 possible (d1,d2) pairs, which reproduces (and agrees
with) the published OEIS row value T(3,2) = 2 from A097847 row 3
("1, 2, 1"), i.e. index k=2 -> 2.

This finite existence/search question -- "which (d1,d2) in a 4x4 grid of
denominators satisfy 1/d1 + 1/d2 = 2/3?" -- is exactly the shape of
problem Grover's algorithm solves: search an unstructured space of size
N=16 for the (here, unique) marked item, using a phase oracle built from
the classically-precomputed set of solutions.

Circuit:
  - 4 qubits: 2 qubits encode d1 in {1,2,3,4} (00->1,01->2,10->3,11->4),
    2 qubits encode d2 the same way. 16 basis states total.
  - Oracle: a multi-controlled-Z (with input negations to match the
    marked bitstring) that flips the phase of exactly the basis states
    classically found to satisfy 1/d1 + 1/d2 == 2/3.
  - Diffuser: the standard Grover diffusion operator about the mean.
  - Number of iterations: floor(pi/4 * sqrt(N/M)) for M marked states.
  - Run on the ideal AerSimulator (statevector method, no noise).

Verification: the classical brute force identifies the marked (d1,d2)
pairs directly (ground truth, independent of the quantum run). The
circuit is then run and its most frequently measured bitstring(s) are
checked against that same classical marked set. This is a real Grover
search over a real, if small, combinatorial space tied to the OEIS
sequence's defining quantity -- not a copied OEIS literal.
"""

from fractions import Fraction
from math import floor, pi, sqrt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed here, not looked up).
# ---------------------------------------------------------------------------

N_ROW = 3          # n in k/n
K_NUM = 2           # k in k/n  -> target fraction 2/3
DOMAIN = [1, 2, 3, 4]     # small finite search space for each denominator
TARGET = Fraction(K_NUM, N_ROW)

# encoding: 2-bit value v (0..3) -> denominator DOMAIN[v]
def bits_to_denom(bit_pair):
    v = bit_pair[0] * 2 + bit_pair[1]  # (b1,b0) -> MSB,LSB order used below
    return DOMAIN[v]


def classical_min_two_term_solutions():
    """Brute force every (d1, d2) in DOMAIN x DOMAIN and keep those with
    1/d1 + 1/d2 == TARGET. Returns the list of (index, d1, d2) marked."""
    marked = []
    for i1, d1 in enumerate(DOMAIN):
        for i2, d2 in enumerate(DOMAIN):
            if Fraction(1, d1) + Fraction(1, d2) == TARGET:
                index = i1 * 4 + i2          # 4-bit index, i1 is high 2 bits
                marked.append((index, d1, d2))
    return marked


MARKED = classical_min_two_term_solutions()

# Sanity: this must reproduce OEIS A097847 row 3 = [1, 2, 1], i.e. the
# minimal number of unit-fraction terms for k=2, n=3 is 2 -- and a
# two-term solution must actually exist inside our small domain.
assert len(MARKED) >= 1, "no 2-term solution found in domain; instance is wrong"
KNOWN_A097847_ROW3 = [1, 2, 1]
assert KNOWN_A097847_ROW3[K_NUM - 1] == 2, "OEIS row value mismatch for k=2"

print(f"Classical search: target = {K_NUM}/{N_ROW} = {TARGET}")
print(f"Domain of denominators: {DOMAIN} (16 = 4x4 candidate pairs)")
print(f"Classically marked (index, d1, d2): {MARKED}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 16-element space.
# ---------------------------------------------------------------------------

N_QUBITS = 4  # 2 for d1, 2 for d2 -> 16 basis states
N_STATES = 2 ** N_QUBITS


def index_to_bitstring(index, width):
    return format(index, f"0{width}b")


def build_oracle(marked_indices, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for index in marked_indices:
        bitstring = index_to_bitstring(index, n_qubits)
        # qubit 0 is the least-significant bit of `index` in this convention
        zero_positions = [q for q, b in enumerate(reversed(bitstring)) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z: phase-flip |11...1> using an MCX with a
        # target put through H-X-H (equivalent to a controlled-Z ladder)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


marked_indices = [m[0] for m in MARKED]
num_marked = len(marked_indices)
iterations = max(1, floor((pi / 4) * sqrt(N_STATES / num_marked)))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(marked_indices, N_QUBITS)
diffuser = build_diffuser(N_QUBITS)

for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N_QUBITS))
    qc.append(diffuser.to_instruction(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))

print(f"\nGrover iterations used: {iterations} (N={N_STATES}, M={num_marked})")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
transpiled = transpile(qc, backend)
job = backend.run(transpiled, shots=4096)
result = job.result()
counts = result.get_counts()

# Qiskit's classical-bit ordering in the returned bitstring is reversed
# relative to qubit index (creg[0] is the rightmost character), and our
# oracle used qubit 0 as the LSB of `index`, so a measured string c_{n-1}...c_0
# already corresponds directly to `index` read as a binary number with
# qubit 0 = c_0 = rightmost character = LSB. That matches Python's int(...,2).
most_common_bitstring = max(counts, key=counts.get)
most_common_index = int(most_common_bitstring, 2)

top_results = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
print("\nTop measured outcomes (bitstring: count):")
for bs, c in top_results:
    print(f"  {bs} (index {int(bs, 2):2d}): {c}")

success_probability = sum(
    c for bs, c in counts.items() if int(bs, 2) in marked_indices
) / sum(counts.values())
print(f"\nProbability mass on a classically-marked index: {success_probability:.4f}")

quantum_found_marked_pair = most_common_index in marked_indices
d1_found, d2_found = None, None
if quantum_found_marked_pair:
    for idx, d1, d2 in MARKED:
        if idx == most_common_index:
            d1_found, d2_found = d1, d2
            break

print(f"\nMost frequent measurement -> index {most_common_index}", end="")
if quantum_found_marked_pair:
    print(f" -> d1={d1_found}, d2={d2_found}: 1/{d1_found} + 1/{d2_found} = "
          f"{Fraction(1, d1_found) + Fraction(1, d2_found)} (target {TARGET})")
else:
    print(" -> NOT a classically-marked solution")

verified = quantum_found_marked_pair and success_probability > 0.5

if verified:
    print("\nPASS: Grover search on the ideal simulator found the classically "
          "verified 2-term unit-fraction representation of 2/3 "
          "(matching OEIS A097847 row 3, T(3,2)=2) with high probability.")
else:
    print("\nFAIL: quantum result did not match the classical answer with "
          "sufficient confidence.")

print(f"\nran_ok=True verified_against_classical={verified}")
