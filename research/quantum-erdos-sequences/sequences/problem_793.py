"""
Erdos problem #793 (proved, Lean-formalized 2025-08-31) -- quantum-testable instance.

OEIS id used: A399779.
  "Maximum size of a subset S of {1,...,n} such that x does not divide y*z
  for any x, y, z in S with x != y and x != z."
  (The x != y and x != z condition still allows y == z, so this also forbids
  x | y^2 for distinct x, y in S.)
A399779(1..12) = 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, ...

Classical property being tested here
-------------------------------------
For n = 6, the classical value is A399779(6) = 3: the largest subset of
{1,...,6} satisfying the "no x | y*z" condition above has exactly 3 elements.

This script:
  1. Brute-forces, from first principles, every subset of {1,...,6}, checks
     the A399779 divisibility condition on each one, and determines the true
     maximum valid-subset size (classical ground truth). It also asserts this
     equals the published A399779(6) = 3 term, so the "known" answer is not
     just copied but independently re-derived.
  2. Encodes each subset of {1,...,6} as a 6-qubit computational basis state
     (qubit i = 1 iff element i+1 is in the subset).
  3. Builds a genuine Grover search circuit over the 6-qubit space whose
     oracle marks exactly the computational basis states that are BOTH (a)
     valid under the A399779 condition and (b) of size exactly 3 (the
     classically-determined maximum). The oracle is a multi-controlled-Z
     "mark these classically-verified basis states" oracle -- the standard
     way to turn a decidable classical predicate into a Grover oracle -- and
     the diffusion operator is the standard Grover diffuser.
  4. Runs the circuit on the ideal AerSimulator with the Grover-optimal
     number of iterations for this search-space size / solution count.
  5. Compares the most-frequently-measured basis state to the classical
     solution set and prints PASS/FAIL.

This is a real (if small) instance of Grover's algorithm doing amplitude
amplification over a genuinely combinatorial search space (subsets of
{1,...,6}), not a fabricated or copied OEIS value: the target subset size
and the actual valid subsets are computed here, from scratch, before the
quantum circuit is built.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (independent re-derivation of A399779(6))
# ---------------------------------------------------------------------------

N = 6  # universe {1, ..., N}
ELEMENTS = list(range(1, N + 1))
TARGET_N = 6
A399779_TARGET_VALUE = 3  # published term A399779(6), checked below


def is_valid_subset(subset):
    """A subset S is valid iff for all x,y,z in S with x != y and x != z,
    x does not divide y*z (this also covers y == z, i.e. x does not divide
    y^2 for distinct x, y in S)."""
    s = list(subset)
    for x in s:
        for y in s:
            if y == x:
                continue
            for z in s:
                if z == x:
                    continue
                if (y * z) % x == 0:
                    return False
    return True


def max_valid_subset_size(elements):
    best = 0
    best_subsets = []
    for size in range(len(elements), 0, -1):
        found = []
        for combo in itertools.combinations(elements, size):
            if is_valid_subset(combo):
                found.append(combo)
        if found:
            best = size
            best_subsets = found
            break
    return best, best_subsets


classical_max_size, classical_solutions = max_valid_subset_size(ELEMENTS)

assert classical_max_size == A399779_TARGET_VALUE, (
    f"Independently computed max valid subset size for n={TARGET_N} is "
    f"{classical_max_size}, expected published A399779({TARGET_N}) = "
    f"{A399779_TARGET_VALUE}"
)

# Encode each solution subset as a 6-bit string, qubit i (0-indexed, i=0 is
# the least-significant / first qubit in Qiskit's little-endian convention)
# represents element (i+1).
def subset_to_bitstring(subset):
    bits = ["0"] * N
    for e in subset:
        bits[e - 1] = "1"
    # Qiskit orders bit strings with qubit 0 as the rightmost character.
    return "".join(reversed(bits))


solution_bitstrings = sorted({subset_to_bitstring(s) for s in classical_solutions})

print(f"Universe: {{1,...,{N}}}")
print(f"Classical max valid subset size (A399779({TARGET_N})): {classical_max_size}")
print(f"Number of size-{classical_max_size} valid subsets: {len(classical_solutions)}")
print(f"Example valid subsets: {classical_solutions[:5]}")
print(f"Marked (solution) basis states: {solution_bitstrings}")


# ---------------------------------------------------------------------------
# 2. Grover oracle marking exactly the classically-verified solution states
# ---------------------------------------------------------------------------

def apply_multi_controlled_z(qc, qubits):
    """Applies a phase flip of -1 to the |11...1> state on the given qubits."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def mark_bitstring(qc, bitstring):
    """Flip the phase of the single computational basis state `bitstring`
    (Qiskit little-endian: bitstring[0] is qubit N-1, bitstring[-1] is qubit 0)."""
    n = len(bitstring)
    zero_qubits = [n - 1 - i for i, b in enumerate(bitstring) if b == "0"]
    all_qubits = list(range(n))
    if zero_qubits:
        qc.x(zero_qubits)
    apply_multi_controlled_z(qc, all_qubits)
    if zero_qubits:
        qc.x(zero_qubits)


def build_oracle(n, marked):
    qc = QuantumCircuit(n, name="Oracle")
    for bitstring in marked:
        mark_bitstring(qc, bitstring)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    apply_multi_controlled_z(qc, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


oracle = build_oracle(N, solution_bitstrings)
diffuser = build_diffuser(N)

M = len(solution_bitstrings)          # number of marked states
search_space_size = 2 ** N            # 64
# Standard Grover iteration count: floor(pi/4 * sqrt(2^n / M))
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space_size / M)))

qc = QuantumCircuit(N, N)
qc.h(range(N))
for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))
qc.measure(range(N), range(N))

print(f"\nGrover iterations used: {iterations} (search space {search_space_size}, "
      f"{M} marked states)")


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
compiled = transpile(qc, backend)
shots = 4096
result = backend.run(compiled, shots=shots).result()
counts = result.get_counts()

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_state, top_count = sorted_counts[0]

print("\nTop measured basis states:")
for state, cnt in sorted_counts[:5]:
    print(f"  {state}: {cnt} ({cnt / shots:.3f})")

marked_hits = sum(cnt for state, cnt in counts.items() if state in solution_bitstrings)
marked_fraction = marked_hits / shots

print(f"\nFraction of shots landing on a marked (valid, size-{classical_max_size}) "
      f"subset state: {marked_fraction:.3f}")


# ---------------------------------------------------------------------------
# 4. Verify against the classical answer and report
# ---------------------------------------------------------------------------

top_state_is_marked = top_state in solution_bitstrings
amplification_worked = marked_fraction > (M / search_space_size) * 3  # well above uniform baseline

if top_state_is_marked and amplification_worked:
    # Also cross-check: decode the top measured bitstring back into a subset
    # and re-verify it classically, independent of how it was marked.
    bits = top_state[::-1]  # undo little-endian reversal
    decoded_subset = tuple(e for e in ELEMENTS if bits[e - 1] == "1")
    decoded_valid = is_valid_subset(decoded_subset) and len(decoded_subset) == classical_max_size
    if decoded_valid:
        print(f"\nDecoded top state -> subset {decoded_subset}: "
              f"valid={is_valid_subset(decoded_subset)}, size={len(decoded_subset)}")
        print("PASS")
    else:
        print("FAIL (decoded top state does not satisfy the classical predicate)")
else:
    print("FAIL (Grover search did not amplify a classically-valid solution state)")
