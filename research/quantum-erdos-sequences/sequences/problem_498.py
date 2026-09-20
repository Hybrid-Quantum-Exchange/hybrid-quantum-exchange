"""
Erdos problem #498 -- quantum-testable instance.

Source metadata (erdosproblems.com data, problems.yaml, number "498"):
  prize: no
  status: proved (Lean), last_update 2026-01-27
  oeis: ["N/A"]   <-- NO OEIS sequence id is attached to this problem.
  tags: ["combinatorics", "analysis"]
  comments: "strong Littlewood-Offord problem"

LIMITATION (reported honestly, per instructions): problem #498 has no OEIS
id in the source data, so there is no genuine "OEIS sequence" to test
membership/growth/divisibility of. There is therefore no way to satisfy the
letter of "identify a property of the OEIS sequence" for this entry. What
follows is the best-effort honest substitute: a small, finite, genuinely
computable combinatorial instance drawn directly from the *mathematical
content* of problem #498 (the Littlewood-Offord problem), verified both
classically (from first principles, in this script) and via a real Grover
search circuit on Qiskit's AerSimulator.

The classical Littlewood-Offord problem: given nonzero reals a_1,...,a_n,
consider the 2^n signed/subset sums sum_{i in S} a_i over all subsets
S subseteq {1,...,n}. The classical Erdos theorem states the number of
subsets S whose sum lands in a fixed open interval of length 2 is at most
C(n, floor(n/2)) -- the central binomial coefficient -- achieved exactly
when all a_i = 1 and the interval is centered so it selects the subsets of
size exactly floor(n/2) (equivalently ceil(n/2), by symmetry of parity).
This central-binomial extremal case is the concrete, small, computable
property we test here: for n = 4 unit coefficients a_i = 1, the number of
subsets S subseteq {1,2,3,4} with sum_{i in S} a_i == 2 is exactly
C(4,2) = 6, and those are exactly the length-4 bitstrings with Hamming
weight 2.

Classical answer (computed here in Python from first principles, not
looked up): enumerate all 16 subsets of {1,2,3,4}, sum each, and count how
many equal 2. That count is compared against C(4,2) via the standard
binomial-coefficient formula, and both must agree with what the quantum
circuit finds.

Quantum construction: a 4-qubit Grover search whose oracle marks exactly
the computational basis states |b3 b2 b1 b0> of Hamming weight 2 (i.e. the
subsets S with |S| = 2, which for unit coefficients is exactly "sum == 2").
This is a genuine oracle built from a real reversible arithmetic circuit
(a Hamming-weight-2 detector built out of Toffoli/X gates on ancillas), not
a hand-picked marked state. Grover amplifies the marked subspace and the
simulator's measurement histogram is compared against the classical
prediction: the 6 weight-2 bitstrings should dominate the distribution.

We verify two things quantumly:
  1. Grover search on the oracle returns, with high probability, a state of
     Hamming weight exactly 2 (a genuine search over the 2^4 = 16 subsets).
  2. The oracle itself, applied to the uniform superposition without any
     Grover amplification (i.e. one pass of "compute oracle, measure
     ancilla, check marked count via repeated shots"), independently lets
     us estimate the number of marked states out of 16 and match it to the
     classical value 6 = C(4,2).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations
from math import comb

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

def classical_littlewood_offord_count(n: int, target: int) -> tuple[int, set]:
    """Enumerate all subsets of {0,...,n-1} with unit coefficients a_i = 1
    and count how many have sum == target. Returns (count, set of subsets
    as frozensets of indices)."""
    marked = set()
    for k in range(n + 1):
        for combo in combinations(range(n), k):
            if len(combo) == target:
                marked.add(frozenset(combo))
    return len(marked), marked


N = 4          # number of coefficients / qubits
TARGET_SUM = 2  # sum_{i in S} 1 == 2  <=>  |S| == 2  (Hamming weight 2)

classical_count, marked_subsets = classical_littlewood_offord_count(N, TARGET_SUM)
classical_binomial = comb(N, N // 2)

assert classical_count == classical_binomial == 6, (
    f"classical sanity check failed: count={classical_count}, "
    f"C(n,floor(n/2))={classical_binomial}"
)

# The marked bitstrings (Qiskit bit order: qubit 0 = least significant /
# rightmost character in the returned bitstring).
marked_bitstrings = set()
for combo in marked_subsets:
    bits = ["0"] * N
    for i in combo:
        bits[N - 1 - i] = "1"
    marked_bitstrings.add("".join(bits))

print(f"Classical: n={N}, target sum={TARGET_SUM}")
print(f"Classical: #subsets with sum==2 is {classical_count} "
      f"(= C({N},{N // 2}) = {classical_binomial})")
print(f"Classical: marked bitstrings = {sorted(marked_bitstrings)}")


# ---------------------------------------------------------------------------
# 2. Oracle: mark exactly the Hamming-weight-2 states on 4 qubits.
#    Built from a real reversible circuit: a "weight == 2" detector using
#    ancilla adders (a small ripple population-count) rather than a
#    hand-picked multi-controlled gate list per marked state.
# ---------------------------------------------------------------------------

def build_weight2_counter(n: int) -> QuantumCircuit:
    """Reversible ripple population counter only (no phase marking): after
    this circuit, qubits [n, n+1] = [c0, c1] hold popcount(data) mod 4."""
    data = list(range(n))
    c0, c1 = n, n + 1
    qc = QuantumCircuit(n + 2, name="popcount")
    for q in data:
        qc.ccx(q, c0, c1)   # carry into c1 first (uses OLD c0)
        qc.cx(q, c0)        # then update c0
    return qc


def full_oracle_and_uncompute(n: int) -> QuantumCircuit:
    """Compute popcount -> mark (CCX phase kickback on weight==2) ->
    uncompute popcount. The marking gate sits BETWEEN compute and its
    inverse, so it survives (it must not be cancelled by the uncompute)."""
    c0, c1 = n, n + 1
    phase = n + 2
    total = n + 3
    counter = build_weight2_counter(n)

    qc = QuantumCircuit(total, name="oracle")
    qc.compose(counter, qubits=range(n + 2), inplace=True)
    # mark weight==2: c1==1 and c0==0
    qc.x(c0)
    qc.ccx(c1, c0, phase)
    qc.x(c0)
    qc.compose(counter.inverse(), qubits=range(n + 2), inplace=True)
    return qc


# ---------------------------------------------------------------------------
# 3. Independent classical check of the oracle logic itself (statevector-free):
#    simulate the tiny reversible counter classically for all 16 inputs and
#    confirm it marks exactly the 6 expected bitstrings, BEFORE trusting it
#    inside a quantum circuit.
# ---------------------------------------------------------------------------

def classical_oracle_check(n: int) -> set:
    marked = set()
    for x in range(2 ** n):
        bits = [(x >> i) & 1 for i in range(n)]  # bits[i] = qubit i
        c0 = c1 = 0
        for q in bits:
            carry = c0 & q
            c1 = c1 ^ carry
            c0 = c0 ^ q
        if c1 == 1 and c0 == 0:
            s = "".join(str(bits[n - 1 - i]) for i in range(n))
            marked.add(s)
    return marked


oracle_marked_check = classical_oracle_check(N)
assert oracle_marked_check == marked_bitstrings, (
    f"oracle logic mismatch: oracle marks {sorted(oracle_marked_check)}, "
    f"expected {sorted(marked_bitstrings)}"
)
print(f"Classical check of oracle's reversible counter logic: PASS "
      f"(marks exactly {sorted(oracle_marked_check)})")


# ---------------------------------------------------------------------------
# 4. Grover search circuit using this oracle, run on AerSimulator.
# ---------------------------------------------------------------------------

def diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def build_grover_circuit(n: int, iterations: int) -> QuantumCircuit:
    n_count, n_phase = 2, 1
    total_ancilla = n_count + n_phase
    total = n + total_ancilla
    phase_idx = n + n_count

    qc = QuantumCircuit(total, n, name="grover")
    qc.h(range(n))          # uniform superposition over 2^n subsets
    qc.x(phase_idx)
    qc.h(phase_idx)         # phase ancilla in |->

    oracle = full_oracle_and_uncompute(n)

    for _ in range(iterations):
        qc.compose(oracle, qubits=range(n + total_ancilla), inplace=True)
        qc.compose(diffuser(n), qubits=range(n), inplace=True)

    qc.h(phase_idx)
    qc.x(phase_idx)
    qc.measure(range(n), range(n))
    return qc


# Optimal Grover iteration count for M=6 marked out of N=16 states.
M, N_STATES = classical_count, 2 ** N
theta = np.arcsin(np.sqrt(M / N_STATES))
optimal_iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
print(f"Grover: M={M} marked out of N={N_STATES}, using {optimal_iterations} iteration(s)")

grover_qc = build_grover_circuit(N, optimal_iterations)

sim = AerSimulator()
compiled = transpile(grover_qc, sim)
shots = 4096
result = sim.run(compiled, shots=shots).result()
counts = result.get_counts()

# Fraction of shots landing on a weight-2 (marked) bitstring.
marked_shots = sum(c for bs, c in counts.items() if bs in marked_bitstrings)
marked_fraction = marked_shots / shots

print(f"Grover: top measured outcomes = "
      f"{sorted(counts.items(), key=lambda kv: -kv[1])[:8]}")
print(f"Grover: fraction of shots landing on a weight-2 (sum==2) subset: "
      f"{marked_fraction:.3f} (uniform-random baseline would be {M / N_STATES:.3f})")

grover_amplified = marked_fraction > 0.7  # Grover should strongly amplify M=6/16 -> high success


# ---------------------------------------------------------------------------
# 5. Independent estimate of |marked states| from one pass of the oracle
#    (no amplification), to cross-check the classical count of 6 without
#    relying on Grover's amplitude amplification at all.
# ---------------------------------------------------------------------------

def build_single_pass_circuit(n: int) -> QuantumCircuit:
    n_count, n_phase = 2, 1
    total_ancilla = n_count + n_phase
    total = n + total_ancilla
    c0_idx, c1_idx = n, n + 1

    qc = QuantumCircuit(total, n_count, name="single_pass")
    qc.h(range(n))
    counter = build_weight2_counter(n)
    qc.compose(counter, qubits=range(n + 2), inplace=True)
    qc.measure(c0_idx, 0)
    qc.measure(c1_idx, 1)
    return qc


single_pass_qc = build_single_pass_circuit(N)
compiled2 = transpile(single_pass_qc, sim)
result2 = sim.run(compiled2, shots=shots).result()
counts2 = result2.get_counts()

# c1 c0 == '10' (Qiskit prints c1 c0 as "c1c0") means weight==2.
weight2_shots = sum(c for bitstr, c in counts2.items() if bitstr == "10")
estimated_fraction = weight2_shots / shots
estimated_count = round(estimated_fraction * N_STATES)

print(f"Single-pass counter estimate: measured weight==2 on "
      f"{estimated_fraction:.3f} of shots -> estimated marked count "
      f"{estimated_count} out of {N_STATES}")

count_matches = estimated_count == classical_count


# ---------------------------------------------------------------------------
# 6. Verdict.
# ---------------------------------------------------------------------------

verified = grover_amplified and count_matches and (oracle_marked_check == marked_bitstrings)

print()
print(f"Classical answer: #{{S subseteq [4] : sum_{{i in S}} 1 == 2}} = {classical_count} "
      f"(= C(4,2))")
print(f"Quantum Grover search amplified success probability to {marked_fraction:.3f} "
      f"(threshold 0.7): {'OK' if grover_amplified else 'FAIL'}")
print(f"Quantum single-pass marked-count estimate: {estimated_count} "
      f"(classical: {classical_count}): {'OK' if count_matches else 'FAIL'}")
print()
print("PASS" if verified else "FAIL")
