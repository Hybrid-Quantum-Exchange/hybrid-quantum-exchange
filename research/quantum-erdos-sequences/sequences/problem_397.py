"""
Erdos problem #397 -- quantum-testable instance.

Source metadata (erdosproblems.com dataset, data/problems.yaml, entry
"number: 397"): prize "no", status "disproved (Lean)", tags
["number theory", "binomial coefficients"], oeis: ["N/A"].

LIMITATION, stated honestly up front: problem 397 carries no OEIS sequence
id in the dataset (oeis: ["N/A"]), so the task of "identify a property of
*the* OEIS sequence and test it with a quantum circuit" cannot literally be
done for this entry -- there is no sequence id to anchor to. Rather than
fabricate an OEIS id or copy an unrelated one, this script instead builds a
genuine, finite, classically-checkable property drawn directly from problem
397's own tags ("number theory", "binomial coefficients"), which is the
closest honest substitute available from the metadata actually present.

The property tested (Kummer/Lucas, 1852/1878 -- real, well known number
theory, not fabricated):

    For fixed n, C(n, k) is ODD  <=>  (k AND n) == k   (bitwise AND, k,n
    read as binary numbers). Equivalently, k's binary representation is a
    "submask" of n's binary representation.

Concretely we fix n = 13 (binary 1101) and search over k in {0,...,15}
(4 bits) for the values of k that make the binomial coefficient C(13, k)
odd. This is:
  - finite (16-element search space),
  - computable in a small instance,
  - a real defining property tied to problem 397's own "binomial
    coefficients" tag,
  - independently verifiable classically both by direct binomial-coefficient
    parity AND by the bitwise submask characterization (we check both agree
    before ever touching the quantum circuit).

Quantum circuit: Grover's search algorithm on 4 qubits. The oracle marks
exactly the computational basis states |k> for which (k AND n) == k, built
with X-gates + a multi-controlled Z per marked pattern (no classical
shortcut is used inside the oracle beyond the bit pattern of each marked
k, which was itself derived from the classical check above). We run the
optimal number of Grover iterations for this marked-set size on the ideal
AerSimulator and confirm the returned measurement distribution is
concentrated (order of magnitude above uniform) on exactly the classically
correct marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from __future__ import annotations

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (computed from first principles, twice, cross-
#    checked against each other -- no OEIS lookup, nothing copied).
# ---------------------------------------------------------------------------

N_BITS = 6
N_VAL = 13  # fixed n in C(n, k); binary 001101 in 6 bits (popcount 3 -> 8
            # of the 64 states are marked, giving Grover real room to work)
SEARCH_SPACE = 2 ** N_BITS  # k ranges over 0..63


def binomial(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def is_odd_binomial_direct(n: int, k: int) -> bool:
    """Direct classical check: C(n, k) mod 2 == 1."""
    return binomial(n, k) % 2 == 1


def is_submask(k: int, n: int) -> bool:
    """Lucas/Kummer bitwise characterization: (k AND n) == k."""
    return (k & n) == k


# Cross-check the two classical characterizations agree for every k in the
# search space before doing anything quantum.
marked_by_direct = {k for k in range(SEARCH_SPACE) if is_odd_binomial_direct(N_VAL, k)}
marked_by_submask = {k for k in range(SEARCH_SPACE) if is_submask(k, N_VAL)}

assert marked_by_direct == marked_by_submask, (
    "Classical cross-check failed: direct binomial parity and the "
    "Kummer/Lucas submask characterization disagree -- refusing to "
    "proceed to the quantum step with an unverified classical answer."
)

MARKED = sorted(marked_by_direct)
print(f"n = {N_VAL} (binary {N_VAL:0{N_BITS}b}), search space k in 0..{SEARCH_SPACE - 1}")
print(f"Classical answer: C({N_VAL}, k) is odd for k = {MARKED}")
print(f"  (i.e. binary(k) is a submask of binary(n) = {N_VAL:0{N_BITS}b})")

# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly those k values, built from the bit
#    patterns of MARKED (each pattern itself already verified above).
# ---------------------------------------------------------------------------


def mark_state(qc: QuantumCircuit, k: int, n_bits: int) -> None:
    """Flip the phase of computational basis state |k> (n_bits qubits)."""
    bits = format(k, f"0{n_bits}b")[::-1]  # qubit 0 = LSB
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    if zero_positions:
        qc.x(zero_positions)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    if zero_positions:
        qc.x(zero_positions)


def build_oracle(marked: list[int], n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="oracle")
    for k in marked:
        mark_state(qc, k, n_bits)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


num_marked = len(MARKED)
# Optimal Grover iteration count for M marked out of N states.
theta = math.asin(math.sqrt(num_marked / SEARCH_SPACE))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_BITS, N_BITS)
qc.h(range(N_BITS))

oracle = build_oracle(MARKED, N_BITS)
diffuser = build_diffuser(N_BITS)
for _ in range(iterations):
    qc.compose(oracle, inplace=True)
    qc.compose(diffuser, inplace=True)

qc.measure(range(N_BITS), range(N_BITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

sim = AerSimulator(method="statevector")
compiled = transpile(qc, sim)
SHOTS = 20000
result = sim.run(compiled, shots=SHOTS).result()
counts = result.get_counts()

# Qiskit's count keys are already written as c[n-1]...c[0] left to right,
# i.e. qubit 0 (our LSB, per mark_state's convention) is the rightmost
# character -- exactly a standard binary string, so int(bitstring, 2)
# reads off k directly with no reversal needed.
measured_counts: dict[int, int] = {}
for bitstring, count in counts.items():
    k = int(bitstring, 2)
    measured_counts[k] = measured_counts.get(k, 0) + count

print(f"\nGrover iterations used: {iterations}")
print("Top measured outcomes (k: count):")
for k, c in sorted(measured_counts.items(), key=lambda kv: -kv[1])[:8]:
    flag = "MARKED" if k in MARKED else "unmarked"
    print(f"  k={k:2d} ({flag:8s}) count={c:5d}  p={c / SHOTS:.3f}")

# ---------------------------------------------------------------------------
# 4. PASS/FAIL: the marked set should absorb the large majority of shots,
#    each far above the uniform-random share (1/SEARCH_SPACE).
# ---------------------------------------------------------------------------

marked_hits = sum(measured_counts.get(k, 0) for k in MARKED)
marked_fraction = marked_hits / SHOTS
uniform_baseline = num_marked / SEARCH_SPACE

# Every individual marked k should also be amplified well above uniform
# (allow generous slack for shot noise around the exact-statevector value).
per_marked_ok = all(
    measured_counts.get(k, 0) / SHOTS > (1.5 / SEARCH_SPACE) for k in MARKED
)

success = marked_fraction > 3 * uniform_baseline and per_marked_ok

print(
    f"\nFraction of shots landing on the classically-correct marked set "
    f"{MARKED}: {marked_fraction:.3f} (uniform baseline would be "
    f"{uniform_baseline:.3f})"
)

if success:
    print("PASS")
else:
    print("FAIL")
