"""
Erdos problem #345 -- quantum-testable instance.

OEIS id used: A001661.
  A001661(n) is the largest positive integer that is NOT expressible as a
  sum of distinct n-th powers of positive integers. (Sprague proved that for
  every n, all sufficiently large integers ARE such a sum; A001661 records
  the largest exception.) For n = 2 this is the classical "sum of distinct
  squares" representability question that problem #345 (tags: "number
  theory", "complete sequences") is about.

Classical property tested here (small, finite, computable):
  Fix the finite set of squares S = {1, 4, 9, 16} (i.e. k^2 for k = 1..4,
  n = 2 as in A001661) and a target integer T = 29. The property is:
      "T is expressible as a sum of a SUBSET of DISTINCT elements of S."
  This is exactly a bounded instance of the representability question that
  A001661 answers in the limit (largest non-representable integer). There
  are 2^4 = 16 candidate subsets, which is small enough to (a) brute-force
  classically in this script and (b) search with a genuine Grover circuit.

  Classical brute force (done in this script, first principles, no OEIS
  value copied) over all 16 subsets of {1, 4, 9, 16} shows exactly one
  subset sums to 29, namely {4, 9, 16} (4 + 9 + 16 = 29), encoded as the
  4-bit string with bit order (b0=1?, b1=4?, b2=9?, b3=16?) -> 0111.

Quantum circuit:
  4 qubits, one per element of S, in uniform superposition over the 16
  subsets. A phase oracle (built directly from the classically computed
  truth table -- i.e. it marks precisely the bitstrings whose subset sums
  equal T, computed by brute force in this script, not hand-picked) is
  combined with the standard Grover diffuser, run for the optimal number of
  Grover iterations for 1 marked item out of 16 (2 iterations). The circuit
  is executed on the ideal AerSimulator and the most frequently measured
  bitstring must decode to the unique subset that classically sums to T.

PASS/FAIL: compare the quantum result (most-sampled bitstring, decoded to a
subset and its sum) against the classical brute-force answer.
"""

from __future__ import annotations

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Problem instance (n = 2, as in A001661) and classical ground truth.
# ---------------------------------------------------------------------------

N_EXPONENT = 2
SQUARES = [k ** N_EXPONENT for k in range(1, 5)]  # [1, 4, 9, 16]
TARGET = 29
NUM_QUBITS = len(SQUARES)


def classical_subset_sums(elements: list[int]) -> dict[str, int]:
    """Brute-force every subset of `elements`, keyed by its bit encoding.

    Bit i (from the least-significant / qubit-0 side) selects elements[i].
    Returns {bitstring: sum} for all 2^len(elements) subsets.
    """
    sums: dict[str, int] = {}
    n = len(elements)
    for bits in itertools.product([0, 1], repeat=n):
        # bits[0] corresponds to qubit 0 (elements[0]), etc.
        total = sum(e for e, b in zip(elements, bits) if b)
        bitstring = "".join(str(b) for b in bits)
        sums[bitstring] = total
    return sums


ALL_SUMS = classical_subset_sums(SQUARES)
MARKED = [bs for bs, s in ALL_SUMS.items() if s == TARGET]

assert len(MARKED) >= 1, (
    f"Instance error: no subset of {SQUARES} sums to {TARGET}; "
    "pick a different TARGET before building the circuit."
)

print(f"Classical brute force over subsets of {SQUARES} (n={N_EXPONENT}):")
print(f"  target T = {TARGET}")
print(f"  marked bitstrings (bit i selects {SQUARES}[i]): {MARKED}")
for bs in MARKED:
    chosen = [SQUARES[i] for i, b in enumerate(bs) if b == "1"]
    print(f"    {bs} -> subset {chosen}, sum = {sum(chosen)}")


# ---------------------------------------------------------------------------
# 2. Grover search circuit over the 2^NUM_QUBITS candidate subsets.
# ---------------------------------------------------------------------------

def build_oracle(num_qubits: int, marked_bitstrings: list[str]) -> QuantumCircuit:
    """Phase oracle flipping the sign of exactly the marked basis states.

    Qiskit's qubit ordering has qubit 0 as the least-significant bit, which
    matches how `classical_subset_sums` assigned bits[0] -> qubit 0.
    """
    qc = QuantumCircuit(num_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        # bitstring[i] is bit i (qubit i). Flip 0-bits to 1 with X so a
        # standard multi-controlled Z fires exactly on this basis state.
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def num_grover_iterations(n_items: int, n_marked: int) -> int:
    if n_marked <= 0:
        return 0
    ratio = math.asin(math.sqrt(n_marked / n_items))
    iterations = math.floor((math.pi / 4) / ratio)
    return max(1, iterations)


oracle = build_oracle(NUM_QUBITS, MARKED)
diffuser = build_diffuser(NUM_QUBITS)

n_items = 2 ** NUM_QUBITS
iterations = num_grover_iterations(n_items, len(MARKED))
print(f"\nGrover search: {n_items} candidates, {len(MARKED)} marked, "
      f"{iterations} iteration(s).")

qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
qc.h(range(NUM_QUBITS))
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)
qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

simulator = AerSimulator()
compiled = transpile(qc, simulator)
result = simulator.run(compiled, shots=4096).result()
counts = result.get_counts()

# Qiskit reports bitstrings as "qN-1 ... q1 q0"; reverse to get our
# (bit0=qubit0 ...) convention used by classical_subset_sums.
def qiskit_key_to_our_bits(key: str) -> str:
    return key[::-1]

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_key, top_count = sorted_counts[0]
top_bits = qiskit_key_to_our_bits(top_key)
top_sum = ALL_SUMS[top_bits]

print("\nMeasurement results (top 5):")
for key, cnt in sorted_counts[:5]:
    bits = qiskit_key_to_our_bits(key)
    print(f"  {key} (bits={bits}, sum={ALL_SUMS[bits]}): {cnt} shots")

print(f"\nMost-sampled outcome decodes to bits={top_bits}, "
      f"subset sum={top_sum}, count={top_count}/{sum(counts.values())} shots")


# ---------------------------------------------------------------------------
# 4. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------------

quantum_found_marked = top_bits in MARKED
classical_says_representable = TARGET in ALL_SUMS.values()

verified = quantum_found_marked and classical_says_representable and (top_sum == TARGET)

print(f"\nClassical: T={TARGET} representable as sum of distinct squares "
      f"from {SQUARES}? {classical_says_representable}")
print(f"Quantum: most-likely Grover outcome is a marked (target-sum) state? "
      f"{quantum_found_marked}")

if verified:
    print("PASS")
else:
    print("FAIL")
