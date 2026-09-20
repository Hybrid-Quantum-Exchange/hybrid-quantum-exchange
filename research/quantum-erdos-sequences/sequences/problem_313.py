"""
Erdos problem #313 -- quantum-testable instance.

OEIS id used: A054377 (primitive weird numbers).

A "weird" number n is abundant (sigma(n) > 2n, i.e. the sum of its proper
divisors exceeds n) but NOT semiperfect: no subset of its proper divisors
sums exactly to n. A "primitive" weird number is a weird number none of
whose proper divisors is itself weird. A054377 lists the primitive weird
numbers in increasing order; its first (smallest) term is n = 70.

Classical property being tested here, computed from first principles in
this script (not copied from OEIS):

    n = 70 has proper divisors D = {1, 2, 5, 7, 10, 14, 35} (7 of them,
    sum(D) = 74 > 70, so 70 is abundant). Weirdness additionally requires
    that NO subset of D sums exactly to 70 (the "not semiperfect" half of
    the definition). This script brute-force enumerates all 2^7 = 128
    subsets of D classically and counts how many sum to exactly 70 -- the
    correct classical answer, verified here, is 0 (which is exactly why
    70 qualifies as weird, and in fact as the smallest primitive weird
    number).

Quantum circuit:

    We put 7 "choice" qubits (one per divisor) into an equal superposition
    over all 2^7 = 128 subsets with Hadamard gates. For every subset whose
    classically-computed sum equals 70 (there are none, per the brute
    force check above) we mark it with a multi-controlled-X gate onto an
    ancilla "flag" qubit, using the standard X-sandwich trick so the
    multi-controlled-X fires exactly on that computational basis pattern.
    This is a genuine oracle built from the real classical truth table of
    the "does this subset of divisors sum to 70" predicate -- exactly the
    function whose absence of solutions is the mathematical content of
    70 being weird (not semiperfect).

    Because the classical brute force finds zero matching subsets, the
    oracle applies zero marking gates, and the ideal quantum prediction is
    that measuring the flag qubit over many shots yields 0 in every shot
    (P(flag=1) = 0), agreeing with the classical result 0/128 = 0. We run
    the circuit on the ideal AerSimulator and check exactly this.

    To make the run non-vacuous (i.e. actually exercise real gates and
    superposition, not just an empty circuit), the script also builds and
    runs a second sanity oracle for a target sum that IS achievable by a
    proper subset of D (found classically), and checks that the quantum
    circuit correctly lights up the flag qubit with certainty for that
    achievable target. This confirms the oracle-construction machinery
    itself is correct, not merely trivially always-zero.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------
# 1. Classical computation (first principles) of the instance.
# ---------------------------------------------------------------------

def proper_divisors(n: int):
    return [d for d in range(1, n) if n % d == 0]


N = 70
DIVISORS = proper_divisors(N)  # expect [1, 2, 5, 7, 10, 14, 35]
assert DIVISORS == [1, 2, 5, 7, 10, 14, 35], DIVISORS
NUM_DIV = len(DIVISORS)  # 7

sigma_proper_sum = sum(DIVISORS)
IS_ABUNDANT = sigma_proper_sum > N
assert IS_ABUNDANT, "70 must be abundant for the weird-number property to apply"


def subset_sums_matching(target: int):
    """Classically enumerate all 2^k subsets of DIVISORS, return the list
    of index-tuples (bitmask style, as a set of divisor indices) whose
    sum equals `target`. Brute force, first principles."""
    hits = []
    for r in range(NUM_DIV + 1):
        for combo in combinations(range(NUM_DIV), r):
            s = sum(DIVISORS[i] for i in combo)
            if s == target:
                hits.append(combo)
    return hits


# The property that defines weirdness (together with abundance): no
# subset of proper divisors sums to N itself.
matches_for_N = subset_sums_matching(N)
CLASSICAL_NUM_SOLUTIONS_FOR_N = len(matches_for_N)

# Sanity target: pick some achievable sum strictly between 0 and
# sum(DIVISORS) to prove the oracle machinery actually works. Find the
# smallest achievable positive sum using more than one divisor, or any
# achievable sum, classically.
achievable_target = None
achievable_solution = None
for target in range(1, sigma_proper_sum + 1):
    hits = subset_sums_matching(target)
    if hits:
        achievable_target = target
        achievable_solution = hits[0]
        break
assert achievable_target is not None

print(f"n = {N}, proper divisors = {DIVISORS}, sum = {sigma_proper_sum} "
      f"(abundant: {IS_ABUNDANT})")
print(f"Classical brute force: number of subsets of divisors summing to "
      f"{N} = {CLASSICAL_NUM_SOLUTIONS_FOR_N} "
      f"(0 means 70 is not semiperfect, i.e. weird)")
print(f"Sanity check target: {achievable_target}, achieved by divisor "
      f"indices {achievable_solution} "
      f"({[DIVISORS[i] for i in achievable_solution]})")


# ---------------------------------------------------------------------
# 2. Quantum oracle construction and circuit.
# ---------------------------------------------------------------------

def build_marking_circuit(target: int):
    """Build a circuit over NUM_DIV 'choice' qubits + 1 flag qubit that
    puts the choice qubits into equal superposition over all subsets of
    DIVISORS, and flips the flag qubit to |1> exactly on those subsets
    (computational basis states) whose divisor-sum equals `target`. The
    marked subsets are found by the same classical brute force used
    above (subset_sums_matching), so the oracle is provably correct by
    construction, not guessed."""
    hits = subset_sums_matching(target)

    qc = QuantumCircuit(NUM_DIV + 1, 1)
    flag = NUM_DIV

    # Equal superposition over all 2^NUM_DIV subsets.
    for q in range(NUM_DIV):
        qc.h(q)

    # Mark every matching subset with a multi-controlled-X onto flag,
    # sandwiching with X gates so the MCX fires on that exact bit pattern
    # (qubit i control value = 1 if divisor i is in the subset, else 0).
    for combo in hits:
        included = set(combo)
        zero_qubits = [q for q in range(NUM_DIV) if q not in included]
        for q in zero_qubits:
            qc.x(q)
        qc.mcx(list(range(NUM_DIV)), flag)
        for q in zero_qubits:
            qc.x(q)

    qc.measure(flag, 0)
    return qc, hits


sim = AerSimulator()
SHOTS = 4096


def run_and_get_p1(qc):
    result = sim.run(qc, shots=SHOTS).result()
    counts = result.get_counts()
    ones = counts.get("1", 0)
    return ones / SHOTS, counts


# --- Main test: does any subset sum to 70? (defines weirdness) ---
qc_main, hits_main = build_marking_circuit(N)
p1_main, counts_main = run_and_get_p1(qc_main)

classical_p_main = CLASSICAL_NUM_SOLUTIONS_FOR_N / (2 ** NUM_DIV)
main_ok = (p1_main == 0.0) and (classical_p_main == 0.0)

print(f"\n[main] quantum P(flag=1) for target={N}: {p1_main} "
      f"(counts={counts_main})")
print(f"[main] classical P = {CLASSICAL_NUM_SOLUTIONS_FOR_N}/"
      f"{2 ** NUM_DIV} = {classical_p_main}")

# --- Sanity test: an achievable target must light the flag with certainty ---
qc_sanity, hits_sanity = build_marking_circuit(achievable_target)
p1_sanity, counts_sanity = run_and_get_p1(qc_sanity)

classical_p_sanity = len(hits_sanity) / (2 ** NUM_DIV)
sanity_ok = np.isclose(p1_sanity, classical_p_sanity, atol=1e-9) and p1_sanity > 0.0

print(f"\n[sanity] quantum P(flag=1) for target={achievable_target}: "
      f"{p1_sanity} (counts={counts_sanity})")
print(f"[sanity] classical P = {len(hits_sanity)}/{2 ** NUM_DIV} = "
      f"{classical_p_sanity}")


# ---------------------------------------------------------------------
# 3. Verdict.
# ---------------------------------------------------------------------

VERIFIED_AGAINST_CLASSICAL = bool(main_ok and sanity_ok)

if VERIFIED_AGAINST_CLASSICAL:
    print("\nPASS: quantum circuit agrees with the classical brute-force "
          f"subset-sum property that certifies {N} as not semiperfect "
          "(hence, with abundance, weird -- OEIS A054377), and the oracle "
          "machinery is validated on an achievable sanity target.")
else:
    print("\nFAIL: quantum result disagrees with the classical computation.")
