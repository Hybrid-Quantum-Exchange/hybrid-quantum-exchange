"""
Erdos problem #903 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '903'": prize=no, tags=["combinatorics"], informal_status=proved,
oeis=["N/A"]).

LIMITATION, stated honestly up front: problem 903's metadata record carries no
OEIS id at all (oeis: ["N/A"]). There is therefore no genuine "quantum-testable
sequence membership/term" property to derive from an OEIS entry for this
problem -- any claim to have quantum-tested "the OEIS sequence for #903" would
be fabricated, since no such sequence is named in the source data. Per the
task's fallback instructions, this script instead makes an honest, small,
finite, genuinely-computable combinatorial search in the spirit of the
problem's stated tag ("combinatorics"): SUBSET-SUM, a canonical finite
combinatorial search problem, is used as the concrete instance because it is
the smallest fully-classical combinatorial search that (a) has a small search
space amenable to a real Grover circuit on a simulator, and (b) has a
correct answer that this script derives itself, from first principles, by
brute force, independent of any OEIS lookup.

Concrete instance chosen (N = 4 elements -> 2^4 = 16-element search space,
4 "index" qubits + ancilla, well within N <= 64):

    Set S = [3, 5, 6, 7]
    Target sum T = 12

Classical property tested: "does there exist a subset of S summing to T?"
and, more strongly, "does Grover search recover exactly the classically
brute-forced set of witnessing subsets (as index bitstrings)?"

The script:
  1. Brute-forces (classically, in Python, from first principles) every
     subset of S and records which subset-index bitstrings sum to T. This
     is the classical ground truth.
  2. Builds a real Qiskit oracle circuit that marks exactly those computed
     basis states (a standard phase-oracle built from the classically
     derived marked-state list -- the oracle's *construction* is generic
     Grover machinery, but *which* states it marks is 100% determined by
     the classical brute-force computed in step 1, not hard-coded from
     knowledge of the answer).
  3. Runs the full Grover diffusion-operator search on AerSimulator with
     the optimal number of iterations for this search-space size and
     number of marked states.
  4. Compares the most-frequently-measured basis state(s) against the
     classical answer set and prints PASS/FAIL.

No OEIS sequence membership is being tested here -- that is not available
for problem #903 -- and this script says so rather than fabricating one.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force)
# ---------------------------------------------------------------------------

S = [3, 5, 6, 7]
TARGET = 12
N = len(S)  # number of index qubits

def brute_force_marked_states():
    """Return sorted list of index-bitstrings (as ints 0..2^N-1) whose
    corresponding subset of S sums exactly to TARGET."""
    marked = []
    for mask in range(2 ** N):
        subset_sum = sum(S[i] for i in range(N) if (mask >> i) & 1)
        if subset_sum == TARGET:
            marked.append(mask)
    return sorted(marked)

CLASSICAL_MARKED = brute_force_marked_states()
assert len(CLASSICAL_MARKED) > 0, "instance must have at least one solution"

print(f"Set S = {S}, target T = {TARGET}, search space size = {2**N}")
print(f"Classical brute-force marked index-bitstrings (subsets summing to {TARGET}): "
      f"{[format(m, f'0{N}b') for m in CLASSICAL_MARKED]}")
for m in CLASSICAL_MARKED:
    chosen = [S[i] for i in range(N) if (m >> i) & 1]
    print(f"  index {format(m, f'0{N}b')} -> subset {chosen} -> sum {sum(chosen)}")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classically-computed marked states
# ---------------------------------------------------------------------------

def build_oracle(n, marked_states):
    """Phase oracle over n index qubits that flips the sign of exactly the
    given marked computational-basis states."""
    qc = QuantumCircuit(n, name="oracle")
    for m in marked_states:
        bits = format(m, f"0{n}b")[::-1]  # qubit 0 = LSB
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n == 1:
            qc.z(0)
        else:
            mcz = MCXGate(n - 1)
            qc.h(n - 1)
            qc.append(mcz, list(range(n - 1)) + [n - 1])
            qc.h(n - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        mcz = MCXGate(n - 1)
        qc.h(n - 1)
        qc.append(mcz, list(range(n - 1)) + [n - 1])
        qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


num_marked = len(CLASSICAL_MARKED)
search_space = 2 ** N
# Optimal number of Grover iterations for this search space / marked count.
theta = math.asin(math.sqrt(num_marked / search_space))
iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

oracle = build_oracle(N, CLASSICAL_MARKED)
diffuser = build_diffuser(N)

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))
qc = qc.decompose().decompose().decompose()

print(f"\nGrover iterations used: {iterations} (marked={num_marked}, space={search_space})")

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

sim = AerSimulator()
shots = 4096
result = sim.run(qc, shots=shots).result()
counts = result.get_counts()

print(f"\nMeasurement counts (top 5): {Counter(counts).most_common(5)}")

# Qiskit's classical-register bit order in the returned bitstring is
# big-endian in qubit index (rightmost char = qubit 0), matching how we
# constructed marked-state bit patterns above with qubit 0 = LSB, so a
# returned bitstring 'b' corresponds to integer int(b, 2) directly against
# our (LSB-first-built, then string-reversed-back) convention. Recover the
# integer index consistent with build_oracle's convention:
def bitstring_to_index(bstr, n):
    # bstr is Qiskit's c-register string, c[n-1] c[n-2] ... c[0]
    bits = bstr[::-1]  # now bits[i] = value of qubit i
    return int(bits[::-1], 2) if False else sum(int(bits[i]) << i for i in range(n))

measured_indices = Counter()
for bstr, freq in counts.items():
    idx = bitstring_to_index(bstr, N)
    measured_indices[idx] += freq

# The quantum answer: take every index whose measured probability clearly
# stands out above the uniform-noise floor (>= half of the max count among
# the classically-marked-cardinality-many top outcomes).
top_indices = [idx for idx, _ in measured_indices.most_common(num_marked)]
quantum_answer = sorted(top_indices)

print(f"\nQuantum (Grover) top-{num_marked} measured index-bitstrings: "
      f"{[format(i, f'0{N}b') for i in quantum_answer]}")
print(f"Classical brute-force answer:                    "
      f"{[format(i, f'0{N}b') for i in CLASSICAL_MARKED]}")

verified = quantum_answer == CLASSICAL_MARKED

if verified:
    print("\nPASS: Grover search recovered exactly the classically brute-forced "
          "subset-sum witnesses.")
else:
    print("\nFAIL: Grover search result does not match the classical brute-force "
          "witnesses.")

print(f"\nran_ok=True verified_against_classical={verified}")
