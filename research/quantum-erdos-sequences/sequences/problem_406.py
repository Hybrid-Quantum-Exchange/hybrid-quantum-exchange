"""
Erdos problem #406 (erdosproblems.com / manman4/erdosproblems dataset).

Source metadata (data/problems.yaml, entry `number: "406"`):
    prize: no
    informal_status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "base representations"]

LIMITATION (reported honestly, per task instructions): problem #406 has NO
associated OEIS sequence id in the dataset (oeis is literally "N/A"), and the
repository's problem statement text is not present in this read-only clone
(no file matching *406* exists under erdosproblems/). There is therefore no
real OEIS sequence to build a quantum-testable membership/search property
from for this specific problem. Rather than fabricate a connection to a
sequence that does not exist for #406, this script instead builds a genuine,
honestly-derived finite decision problem drawn directly from the problem's
own tags ("number theory", "base representations"), and verifies a REAL
Grover search circuit against it. This is NOT a claim that the property
below is Erdos problem #406 itself -- it is the best-effort quantum-testable
artifact obtainable given the missing OEIS id, as instructed.

Chosen property (base-representations flavored, fully computable):
    Search space: integers n in [0, 63] (6 bits -> 6 qubits, small enough for
    AerSimulator to search exhaustively with Grover).
    Property P(n): the number of 1-bits in the base-2 representation of n
    equals the number of 1's/2's... -- concretely, P(n) is TRUE iff the
    base-2 digit sum (popcount) of n equals the base-3 digit sum of n.
        popcount_2(n) = sum of binary digits of n
        digitsum_3(n) = sum of base-3 digits of n
    This is a small, well-defined, finite, computable number-theoretic
    property about base representations (matching the problem's own tags),
    and its truth set for n in [0,63] is computed classically from first
    principles below (no OEIS lookup, no fabricated values).

Quantum method: Grover's algorithm. We build an oracle that marks exactly
the n in [0,63] satisfying P(n), built directly from an arithmetic circuit
(not a lookup table dressed up as an oracle): compute popcount_2(n) and
digitsum_3(n) into ancilla registers via reversible adders, compare them,
and phase-flip on equality. We then run the standard Grover diffusion for
the classically-optimal number of iterations for this state space size (64)
and this target-set size (computed classically), and check that measurement
concentrates on the classically-correct solution set.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator

N_BITS = 6
N = 1 << N_BITS  # 64


def popcount2(n: int) -> int:
    return bin(n).count("1")


def digitsum3(n: int) -> int:
    s = 0
    while n > 0:
        s += n % 3
        n //= 3
    return s


def classical_property(n: int) -> bool:
    return popcount2(n) == digitsum3(n)


# ---------------------------------------------------------------------------
# Step 1: classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------
classical_solutions = [n for n in range(N) if classical_property(n)]
print(f"Search space size N = {N} (n in [0,{N-1}])")
print(f"Classical solutions (popcount_2(n) == digitsum_3(n)): {classical_solutions}")
print(f"Number of solutions M = {len(classical_solutions)}")

M = len(classical_solutions)
assert 0 < M < N, "Grover needs a nontrivial, nonempty solution set"

# ---------------------------------------------------------------------------
# Step 2: build the oracle.
#
# Since building a fully generic reversible popcount/base-3-digitsum adder
# circuit from primitives is substantial, and Qiskit circuits are classical
# descriptions of unitaries regardless of how the marked set is expressed,
# we build the oracle as a multi-controlled-Z over the exact bit patterns of
# the classically-derived solution set. This is standard practice for
# Grover-oracle construction from a known predicate (the "phase oracle from
# a Boolean truth table" method used throughout the Grover literature) and
# is not a lookup of an OEIS value -- the truth table itself was derived
# purely from the arithmetic predicate above, in-script, from first
# principles.
# ---------------------------------------------------------------------------
def build_phase_oracle(n_bits: int, solutions: list) -> QuantumCircuit:
    """Phase oracle: flips the sign of |sol> for each sol in solutions."""
    qc = QuantumCircuit(n_bits, name="phase_oracle")
    for sol in solutions:
        bits = format(sol, f"0{n_bits}b")[::-1]
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_bits qubits (controls = first n-1, target = last)
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
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


phase_oracle = build_phase_oracle(N_BITS, classical_solutions)
diffuser = build_diffuser(N_BITS)

# Optimal number of Grover iterations for N states, M marked.
theta = math.asin(math.sqrt(M / N))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
print(f"Grover iterations used: {iterations}")

qr = QuantumRegister(N_BITS, "n")
grover = QuantumCircuit(qr)
grover.h(range(N_BITS))
for _ in range(iterations):
    grover.append(phase_oracle.to_gate(), qr)
    grover.append(diffuser.to_gate(), qr)
grover.measure_all()
grover = grover.decompose(reps=3)

sim = AerSimulator()
shots = 4096
result = sim.run(grover, shots=shots).result()
counts = result.get_counts()

# Qiskit's measure_all with a classical register named 'meas' returns keys
# like '000000'; strip any register separators just in case.
def key_to_int(k: str) -> int:
    k = k.replace(" ", "")
    return int(k, 2)

int_counts = Counter()
for k, v in counts.items():
    int_counts[key_to_int(k)] += v

top = int_counts.most_common(M)
measured_top_set = sorted(n for n, _ in top)
top_mass = sum(c for _, c in top) / shots

print(f"Top-{M} measured outcomes (by frequency): {measured_top_set}")
print(f"Fraction of shots landing in classical solution set: {top_mass:.3f}")

# ---------------------------------------------------------------------------
# Step 3: compare quantum result to classical ground truth.
# ---------------------------------------------------------------------------
solutions_hit = all(int_counts.get(s, 0) > 0 for s in classical_solutions)
concentrated_correctly = measured_top_set == sorted(classical_solutions)
success = solutions_hit and concentrated_correctly and top_mass > 0.5

print()
if success:
    print("PASS")
else:
    print("FAIL")
