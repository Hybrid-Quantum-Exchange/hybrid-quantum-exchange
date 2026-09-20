"""
Erdos problem #696 (per erdosproblems.com metadata, data/problems.yaml).

Metadata for #696: prize "no", status "solved (Lean)", tags
["number theory", "divisors"], oeis: ["possible"].

IMPORTANT LIMITATION: the metadata does not give a concrete OEIS sequence id
for #696 -- the oeis field is the literal placeholder string "possible", not
an id such as "A000005". There is therefore no specific published integer
sequence to target directly. Rather than fabricate an OEIS id, this script
targets the one concrete, finite, computable mathematical object the
metadata *does* commit to: the "divisors" tag, via the classical divisor-
counting function d(n) (OEIS A000005, the divisor-count function itself,
which is unambiguous and is exactly what "number theory, divisors" refers
to), restricted to a small finite search space.

Classical property tested
--------------------------
Search space: integers n in [1, N] with N = 15 (fits in 4 qubits, since
2**4 = 16 >= 15+1 states, state 0 is unused/padding).

Property: n has exactly k = 4 divisors (d(n) == 4), i.e. n is of the form
p^3 or p*q for distinct primes p, q. This is computed here from first
principles (trial division), not copied from OEIS.

For N = 15 this is computed below; as of this script's own computation the
marked set is {6, 8, 10, 14, 15} (n=6=2*3, 8=2^3, 10=2*5, 14=2*7, 15=3*5;
excluded: 12 has 6 divisors).

Quantum approach
-----------------
Grover's search algorithm on 4 qubits over the universe {0,...,15}, with a
multi-controlled-Z oracle built directly from the classically-computed
marked set (a standard "known target set" Grover oracle -- the oracle is
honest because the marked bitstrings are derived from the classical
computation above, not asserted from any external source). One Grover
iteration (optimal for ~5 marked out of 16) is run on the ideal AerSimulator,
and the most-probable measured outcomes are compared against the classical
marked set.

PASS criterion: every one of the top-len(marked_set) most frequent measured
outcomes decodes to an integer in the classically-computed marked set.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the property, from first principles.
# ---------------------------------------------------------------------------

def num_divisors(n: int) -> int:
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


N = 15          # search space is {1, ..., 15}, encoded in 4 qubits (0..15)
K = 4           # target divisor count

classical_marked = sorted(n for n in range(1, N + 1) if num_divisors(n) == K)
print(f"Classical: integers in [1,{N}] with exactly {K} divisors = {classical_marked}")
assert classical_marked == [6, 8, 10, 14, 15], (
    f"classical computation changed unexpectedly: {classical_marked}"
)

NUM_QUBITS = 4
assert N < 2 ** NUM_QUBITS


# ---------------------------------------------------------------------------
# 2. Build a Grover oracle marking exactly the classically-computed states.
# ---------------------------------------------------------------------------

def apply_multi_controlled_z_on_bitstring(qc: QuantumCircuit, bits: str) -> None:
    """Flip the sign of |bits> (a NUM_QUBITS-length '0'/'1' string, MSB..LSB
    matching qc.qubits[0]..qc.qubits[-1]) via X-sandwiched multi-controlled Z."""
    n = len(bits)
    flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
    for i in flip_qubits:
        qc.x(i)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    for i in flip_qubits:
        qc.x(i)


def oracle(qc: QuantumCircuit, marked_values: list) -> None:
    for v in marked_values:
        bits = format(v, f"0{NUM_QUBITS}b")
        apply_multi_controlled_z_on_bitstring(qc, bits)


def diffuser(qc: QuantumCircuit) -> None:
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    qc.h(NUM_QUBITS - 1)
    qc.mcx(list(range(NUM_QUBITS - 1)), NUM_QUBITS - 1)
    qc.h(NUM_QUBITS - 1)
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))


def build_grover_circuit(marked_values: list) -> QuantumCircuit:
    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))

    M = len(marked_values)
    Nstates = 2 ** NUM_QUBITS
    iterations = max(1, round((math.pi / 4) * math.sqrt(Nstates / M)))

    for _ in range(iterations):
        oracle(qc, marked_values)
        diffuser(qc)

    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

qc = build_grover_circuit(classical_marked)

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical bit order in the count keys is c[NUM_QUBITS-1] ... c[0],
# and c[i] was filled from qubit i, which we treated as MSB..LSB matching the
# bitstrings used to build the oracle -- so reverse the returned key to match.
decoded_counts = Counter()
for bitstring, freq in counts.items():
    reordered = bitstring[::-1]  # now qubit0(bit0=MSB) .. qubit(NUM_QUBITS-1)
    value = int(reordered, 2)
    decoded_counts[value] += freq

top = decoded_counts.most_common(len(classical_marked))
top_values = sorted(v for v, _ in top)

print(f"Quantum (Grover, {shots} shots): top {len(classical_marked)} measured "
      f"values = {top_values}")
print(f"Full decoded histogram: {dict(sorted(decoded_counts.items()))}")

quantum_matches_classical = top_values == classical_marked

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
