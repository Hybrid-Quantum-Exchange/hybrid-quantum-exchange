"""
Erdos Problem #421 -- quantum-testable instance
=================================================

Erdos problem: https://www.erdosproblems.com/421
"Is there a sequence 1 <= d_1 < d_2 < ... with density 1 such that all
products prod_{u<=i<=v} d_i (products of consecutive runs of terms) are
distinct?"  (Resolved affirmatively; Selfridge's greedy construction, later
completed by Tao et al.)

OEIS ids used:
  - A389544: the greedy witnessing sequence itself. a(1) = 2; a(n) is the
    smallest integer greater than a(n-1) such that every product of a
    contiguous run of terms in a(1..n) is distinct from every other such
    product.
  - A390848: complement of A389544 in the positive integers.

Classical property tested (computed from first principles in this script,
not copied from OEIS):
  For n = 1..32, is n a member of A389544?  Equivalently: is n a term the
  greedy algorithm keeps (member), or does including it force a repeated
  consecutive-run product, so it is skipped (a term of the complement,
  A390848)?

  The script first runs the literal greedy construction (build up the
  sequence term by term, checking every consecutive-run product for
  collisions) far enough to cover 1..40, and records which of 1..32 are
  MEMBERS vs NON-MEMBERS.  This reproduces, from scratch, the classical
  answer:
      members<=32    = {2,3,4,5,7,8,9,10,11,13,14,15,17,18,19,21,22,23,
                         25,26,27,28,29,30,31,32}
      non-members<=32 = {1,6,12,16,20,24}
  (Cross-checked: 1,6,12,16,20,24 also matches the start of A390848 as
  fetched from OEIS -- but the classical computation below is what the
  quantum circuit is actually checked against, not the fetched values.)

Quantum circuit:
  A genuine Grover search over the 5-qubit register encoding n-1 for
  n = 1..32 (N = 32 basis states).  The oracle is built classically from
  the greedy computation above and marks exactly the 6 non-member indices
  {1,6,12,16,20,24}.  Grover's algorithm amplifies those 6 marked basis
  states; after the classically-optimal number of iterations
  (round(pi/4 * sqrt(N/M))) the AerSimulator statevector should place
  (almost) all probability mass on the 6 marked computational basis
  states, and negligible mass elsewhere.

  The script runs the circuit, reads off the set of basis states whose
  measured probability clears a threshold, maps them back to integers,
  and compares that set to the classically computed non-member set.
  PASS if they match exactly (with high per-shot concentration on the
  marked states); FAIL otherwise.

No fabrication: the "known term" fact used is that 1, 6, 12, 16, 20, 24 are
the first six non-members of the greedy sequence -- and that fact is
re-derived here by direct simulation of the greedy rule, not asserted from
memory or copied from an OEIS listing.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


# ---------------------------------------------------------------------
# 1. Classical computation: build A389544 by the literal greedy rule,
#    from first principles, and derive membership for n = 1..32.
# ---------------------------------------------------------------------

def has_repeated_consecutive_product(seq):
    """True iff some two contiguous runs of `seq` share the same product."""
    n = len(seq)
    seen = set()
    for i in range(n):
        p = 1
        for j in range(i, n):
            p *= seq[j]
            if p in seen:
                return True
            seen.add(p)
    return False


def build_A389544(limit_value):
    """Greedy construction of A389544 up to (and a bit past) limit_value."""
    seq = [2]
    cur = 2
    while cur < limit_value + 8:  # margin so we know true membership <= limit_value
        cur += 1
        trial = seq + [cur]
        if not has_repeated_consecutive_product(trial):
            seq.append(cur)
    return seq


N_RANGE = 32  # test n = 1 .. 32  (fits exactly in 5 qubits: 2**5 = 32)
NUM_QUBITS = 5

full_seq = build_A389544(N_RANGE)
members = sorted(x for x in full_seq if x <= N_RANGE)
non_members = sorted(x for x in range(1, N_RANGE + 1) if x not in members)

print("Classical A389544 members in 1..%d:     %s" % (N_RANGE, members))
print("Classical A389544 non-members in 1..%d: %s" % (N_RANGE, non_members))

# indices (n-1) of the non-members -- these are the Grover "marked" states
marked_indices = sorted(n - 1 for n in non_members)
M = len(marked_indices)
assert M > 0

print("Marked indices (n-1) for Grover oracle:", marked_indices)


# ---------------------------------------------------------------------
# 2. Quantum circuit: Grover search for the non-member indices.
# ---------------------------------------------------------------------

def build_oracle(num_qubits, marked_indices):
    """Phase-flip oracle marking each index in `marked_indices`."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for idx in marked_indices:
        bits = format(idx, "0%db" % num_qubits)  # MSB..LSB over qubits[n-1..0]
        # flip qubits that should be 0 so the target pattern becomes |11..1>
        zero_qubits = [num_qubits - 1 - i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_circuit(num_qubits, marked_indices, iterations):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(num_qubits, marked_indices)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


N_total = 2 ** NUM_QUBITS
theta = math.asin(math.sqrt(M / N_total))
# Choose the iteration count (over a small practical range) that maximizes
# the theoretical success probability sin((2k+1)*theta)^2, rather than
# trusting the coarse round(pi/4*sqrt(N/M)) estimate, which is inaccurate
# for small N/M.
candidate_ks = range(1, 6)
optimal_iters = max(candidate_ks, key=lambda k: math.sin((2 * k + 1) * theta) ** 2)
print("Grover iterations used:", optimal_iters,
      "(theoretical success prob %.4f)" % math.sin((2 * optimal_iters + 1) * theta) ** 2)

qc = grover_circuit(NUM_QUBITS, marked_indices, optimal_iters)

simulator = AerSimulator()
compiled = transpile(qc, simulator)
shots = 20000
result = simulator.run(compiled, shots=shots).result()
counts = result.get_counts()

# ---------------------------------------------------------------------
# 3. Compare quantum result to the classical answer.
# ---------------------------------------------------------------------

# Qiskit bit ordering: classical bit c[i] <- qubit i, and the returned
# bitstring is written c[n-1]...c[1]c[0]  (little-endian display).
# Our oracle encoded qubit (num_qubits-1-i) <-> bit i of `idx` in the
# standard format(idx, '0{n}b') MSB-first string, i.e. qubit j holds bit
# (num_qubits-1-j) of idx (MSB on the highest-index qubit). Qiskit prints
# bitstrings as c[n-1]...c[0], which is exactly that MSB-first order, so
# int(bitstring, 2) recovers idx directly.
measured_indices = Counter()
for bitstring, cnt in counts.items():
    idx = int(bitstring, 2)
    measured_indices[idx] += cnt

total_shots = sum(measured_indices.values())
sorted_hits = sorted(measured_indices.items(), key=lambda kv: -kv[1])

print("\nTop measured indices (index -> count):")
for idx, cnt in sorted_hits[:10]:
    n_value = idx + 1
    print("  n=%2d (idx=%2d): %5d shots (%.1f%%)%s" % (
        n_value, idx, cnt, 100.0 * cnt / total_shots,
        "  <- marked" if idx in marked_indices else ""))

# Take the M most frequently measured indices as the circuit's answer set.
top_M = [idx for idx, _ in sorted_hits[:M]]
found_non_members = sorted(idx + 1 for idx in top_M)

# Require that essentially all probability mass sits on the marked states.
marked_mass = sum(cnt for idx, cnt in measured_indices.items() if idx in marked_indices)
marked_fraction = marked_mass / total_shots

print("\nClassical non-members (n):", non_members)
print("Quantum top-%d measured n :" % M, found_non_members)
print("Fraction of shots landing on a marked state: %.4f" % marked_fraction)

sets_match = set(found_non_members) == set(non_members)
high_concentration = marked_fraction > 0.90  # Grover should heavily amplify the marked set

verified = sets_match and high_concentration

print("\nRESULT:", "PASS" if verified else "FAIL")
if not verified:
    raise SystemExit(1)
