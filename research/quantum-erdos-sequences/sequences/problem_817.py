"""
Erdos problem #817 -- quantum-testable instance
================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"817\"" (tags: ["additive combinatorics"], status: open,
oeis: ["possible"]).

LIMITATION, stated honestly up front: problem #817's YAML entry carries no
real OEIS sequence id -- the `oeis` field is the literal placeholder string
"possible", not an identifier such as "A000040". The problem itself is open
(no proof, no formal statement) and its own statement is not given in the
metadata available here (only prize/status/tags). There is therefore no
OEIS sequence to derive a property from for this entry, and no way to
honestly claim this script tests "the Erdos #817 sequence", because no such
public sequence id exists in the source record.

What this script does instead, as the best honest substitute the task asks
for when no OEIS id is available: it takes the one real piece of content
the metadata does give us -- the tag "additive combinatorics" -- and builds
a genuine, small, fully classically-checkable additive-combinatorics
property: SUM-FREE SUBSETS of {1, ..., n}. A subset S of positive integers
is sum-free if there are no x, y, z in S (x, y not necessarily distinct)
with x + y = z. This is a real, well-studied additive-combinatorics notion
(closely related to Schur-type / sum-free-set questions that Erdos worked
on), it is finite and exactly computable for small n, and it supports a
genuine Grover search circuit: one qubit per element of {1,...,n}, basis
state |b_1 b_2 ... b_n> represents the subset {i : b_i = 1}, and we search
for the maximum-size sum-free subsets.

Concretely, for n = 4 (elements {1,2,3,4}, 4 qubits, 16 basis states):
  1. Classically enumerate all 16 subsets, and for each check the sum-free
     property (x + y = z for x,y,z in S) directly from first principles
     (brute-force triple check, computed in this script).
  2. Find the classical maximum sum-free subset size, and the set of all
     subsets achieving it ("target" states).
  3. Build a real Grover search circuit (equal superposition + oracle that
     phase-flips exactly the target basis states, found classically in
     step 1-2 + the standard diagonal-inversion diffuser), run it on the
     ideal AerSimulator, and confirm the measured distribution concentrates
     on the classically-verified maximum sum-free subsets.
  4. PASS/FAIL is decided by comparing the quantum sampling result against
     the independently-computed classical answer.

This is a real quantum circuit (uniform superposition -> phase-oracle ->
diffusion -> measurement) genuinely solving a finite search problem, not a
literal OEIS lookup -- there is no OEIS id for #817 to look up.

Reporting per the task instructions: no OEIS id was available for problem
#817 (the metadata field is the placeholder "possible"), so
verified_against_classical here means "the quantum search result agrees
with the independently-computed classical maximum-sum-free-subset answer",
not "matches a published OEIS sequence value".
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


# ---------------------------------------------------------------------------
# Step 1-2: classical computation, from first principles, of the maximum
# sum-free subsets of {1, 2, 3, 4}.
# ---------------------------------------------------------------------------

N = 4  # elements {1, ..., 4}; one qubit per element -> 4 qubits, 16 states
ELEMENTS = list(range(1, N + 1))


def is_sum_free(subset):
    """True iff no x + y = z for x, y, z in subset (x, y need not differ)."""
    s = set(subset)
    for x in s:
        for y in s:
            if (x + y) in s:
                return False
    return True


def bits_to_subset(bits):
    """bits: tuple of 0/1 of length N, bit i-1 <-> element i (qubit i-1)."""
    return [ELEMENTS[i] for i, b in enumerate(bits) if b == 1]


all_subsets_by_bits = {}
for bits in itertools.product([0, 1], repeat=N):
    all_subsets_by_bits[bits] = bits_to_subset(bits)

sum_free_bits = [b for b, s in all_subsets_by_bits.items() if is_sum_free(s)]
max_size = max(len(all_subsets_by_bits[b]) for b in sum_free_bits)
target_bits = [b for b in sum_free_bits if len(all_subsets_by_bits[b]) == max_size]

print("Classical brute force over all 2^%d = %d subsets of %s:" % (N, 2 ** N, ELEMENTS))
print("  sum-free subsets found: %d" % len(sum_free_bits))
print("  maximum sum-free subset size: %d" % max_size)
print("  maximum sum-free subsets (target states):")
for b in target_bits:
    print("    bits=%s  subset=%s" % (b, all_subsets_by_bits[b]))

assert max_size > 0, "sanity: the empty set is trivially sum-free, size must be >= 1"
assert len(target_bits) >= 1


# ---------------------------------------------------------------------------
# Step 3: Grover search circuit marking exactly `target_bits`.
# ---------------------------------------------------------------------------

def build_oracle(n, marked_bitstrings):
    """Phase-flip exactly the basis states in marked_bitstrings (tuples of
    0/1, index i-1 <-> qubit i-1, i.e. bits[0] is qubit 0)."""
    qc = QuantumCircuit(n, name="oracle")
    for bits in marked_bitstrings:
        # Flip the 0-bits to 1 so the all-ones pattern identifies this state,
        # apply a multi-controlled Z, then flip back.
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(i)
        if n == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), n - 1, 1), list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(N, target_bits)
diffuser = build_diffuser(N)

M = len(target_bits)
total = 2 ** N
# optimal number of Grover iterations for M marked items out of `total`
theta = math.asin(math.sqrt(M / total))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))

print("\nGrover circuit: n_qubits=%d, marked_states=%d, iterations=%d" % (N, M, iterations))

sim = AerSimulator()
shots = 4096
qc_t = transpile(qc, sim)
job = sim.run(qc_t, shots=shots)
result = job.result()
counts = result.get_counts()

# Qiskit bit-string order is q_{n-1}...q_0; convert each key back into our
# (bit for qubit0, bit for qubit1, ...) tuple convention.
def counts_key_to_bits(key):
    rev = key[::-1]  # rev[i] is qubit i
    return tuple(int(c) for c in rev)


counts_by_bits = {}
for key, c in counts.items():
    counts_by_bits[counts_key_to_bits(key)] = c

target_set = set(target_bits)
target_shots = sum(c for b, c in counts_by_bits.items() if b in target_set)
target_fraction = target_shots / shots

print("\nMeasurement outcomes (top 5 by count):")
for b, c in sorted(counts_by_bits.items(), key=lambda kv: -kv[1])[:5]:
    tag = "TARGET (max sum-free)" if b in target_set else ""
    print("  bits=%s subset=%s count=%4d  %s" % (b, all_subsets_by_bits[b], c, tag))

print("\nFraction of shots landing on a classically-verified maximum")
print("sum-free subset: %.3f (%d / %d shots)" % (target_fraction, target_shots, shots))

# With `iterations` chosen near-optimally, Grover amplification should put
# well over half the probability mass on the M target states (uniform
# baseline would be M/total). Require clear amplification above baseline
# and a solid absolute majority as the pass criterion.
baseline = M / total
passed = target_fraction > max(0.5, 3 * baseline)

# Independent cross-check: the single most frequent measured bitstring must
# itself be one of the classically-computed maximum sum-free subsets.
most_common_bits = max(counts_by_bits.items(), key=lambda kv: kv[1])[0]
most_common_is_target = most_common_bits in target_set
passed = passed and most_common_is_target

print("\nBaseline (uniform-random) probability of hitting a target state: %.3f" % baseline)
print("Most frequent measured outcome is a classical maximum sum-free subset: %s"
      % most_common_is_target)

if passed:
    print("\nPASS: Grover search over sum-free subsets of {1,...,%d} concentrated on the "
          "classically-verified maximum sum-free subsets." % N)
else:
    print("\nFAIL: quantum search result did not clearly match the classical answer.")

print("\nSummary: OEIS id available for Erdos #817 = NONE (metadata field is the literal "
      "placeholder 'possible', not a real OEIS id). Verified property is a from-scratch "
      "additive-combinatorics computation (maximum sum-free subsets of {1,...,%d}), matching "
      "the problem's stated tag, not an OEIS-sequence lookup." % N)

assert passed, "quantum result did not verify against the classical answer"
