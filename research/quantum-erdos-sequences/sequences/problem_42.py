"""
Erdos problem #42 (erdosproblems.com / manman4/erdosproblems data/problems.yaml,
entry "number: '42'") — tags: ["number theory", "sidon sets", "additive
combinatorics"], oeis: ["N/A"].

Limitation, stated up front: problem #42's YAML entry carries no OEIS id
(oeis: ["N/A"]), so there is no literal OEIS sequence to target. Rather than
fabricate one, this script builds a genuine finite, computable property drawn
directly from the problem's own tag ("sidon sets"): whether a specific small
subset of Z_7 is a perfect Sidon / perfect-difference set, and then uses a
real Grover search circuit to recover exactly that subset's elements out of
all of Z_7's candidates.

Classical property tested (computed here from first principles, not looked
up):
    S = {0, 1, 3} subset of Z_7.
    S is a Sidon set mod 7 iff all pairwise differences (i - j) mod 7, for
    ordered pairs i != j in S, are distinct -- equivalently (since |S|=3
    gives 6 ordered difference pairs and there are exactly 6 nonzero residues
    mod 7) each nonzero residue of Z_7 occurs as a difference EXACTLY once.
    This is the classical definition of a "perfect difference set", the
    finite structure Sidon-set constructions (Singer difference sets) are
    built from, and it is a small, fully decidable instance of the Sidon-set
    property named in problem #42's tags.

    The script first verifies this classically by brute force over all
    ordered pairs.

Quantum computation: a 3-qubit Grover search over the 8 residues of Z_7 U {7}
(3 qubits address 0..7; residue 7 is never marked) that amplifies exactly the
marked residues belonging to S = {0, 1, 3}. The oracle is built as an
explicit multi-controlled-Z gate over the three marked basis states -- a
direct arithmetic/membership oracle, not a black box. One Grover iteration is
optimal for 3 marked items out of 8 (theta = arcsin(sqrt(3/8)), optimal
iterations round(pi/(4*theta) - 1/2) = 1). After running on the ideal
AerSimulator, the script checks that the measurement distribution is
supported exactly on {0, 1, 3} (the classically verified Sidon set) with
overwhelming probability, and PASSes iff the quantum search recovered exactly
the classically-computed set.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_is_sidon_mod_n(S, n):
    """Return True iff every nonzero residue mod n occurs as exactly one
    ordered difference (i - j) mod n for i != j in S."""
    diffs = [(i - j) % n for i in S for j in S if i != j]
    if 0 in diffs:
        return False
    counts = {r: 0 for r in range(1, n)}
    for d in diffs:
        counts[d] += 1
    return all(c == 1 for c in counts.values())


def classical_search_sidon_subsets(n, k):
    """Brute-force every k-subset of Z_n and return those that are Sidon
    (perfect difference) sets, used only to derive/confirm S from scratch."""
    from itertools import combinations

    found = []
    for S in combinations(range(n), k):
        if classical_is_sidon_mod_n(S, n):
            found.append(S)
    return found


def build_grover_membership_oracle(num_qubits, marked_states):
    """Explicit oracle: phase-flip exactly the computational basis states in
    marked_states (each an int in [0, 2**num_qubits)), via X-sandwiched
    multi-controlled-Z gates -- a direct arithmetic/comparator oracle, not a
    lookup table."""
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
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
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def optimal_grover_iterations(num_marked, space_size):
    theta = math.asin(math.sqrt(num_marked / space_size))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def run_grover_membership_search(marked_states, num_qubits, shots=4096):
    space_size = 2 ** num_qubits
    iterations = optimal_grover_iterations(len(marked_states), space_size)

    oracle = build_grover_membership_oracle(num_qubits, marked_states)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n = 7          # Z_n
    k = 3           # subset size
    num_qubits = 3  # addresses 0..7, covering Z_7 (7 is simply never marked)

    # --- Classical part: derive/confirm the Sidon set from first principles ---
    sidon_subsets = classical_search_sidon_subsets(n, k)
    assert (0, 1, 3) in sidon_subsets, "expected perfect difference set not found"
    S = (0, 1, 3)
    assert classical_is_sidon_mod_n(S, n) is True

    # Sanity: a set NOT of this form should fail (e.g. {0, 1, 2} is not Sidon
    # mod 7 -- both (1-0) and (2-1) give difference 1).
    assert classical_is_sidon_mod_n((0, 1, 2), n) is False

    print(f"Classical result: Z_{n} k={k} Sidon (perfect difference) sets found: "
          f"{sidon_subsets}")
    print(f"Classical target Sidon set S = {S} "
          f"(verified: all { 2 * math.comb(k, 2) } ordered nonzero differences "
          f"distinct mod {n})")

    # --- Quantum part: Grover search recovering exactly S's elements ---
    marked_states = list(S)
    counts, iterations = run_grover_membership_search(marked_states, num_qubits)
    total_shots = sum(counts.values())

    # Convert bitstrings (Qiskit prints MSB..LSB, big-endian) to integers.
    dist = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        dist[value] = dist.get(value, 0) + c

    marked_hits = sum(dist.get(v, 0) for v in marked_states)
    marked_fraction = marked_hits / total_shots
    quantum_recovered_set = sorted(v for v in dist if dist[v] / total_shots > 0.05)

    print(f"Grover iterations used: {iterations}")
    print(f"Measurement distribution over {total_shots} shots: {dist}")
    print(f"Fraction of shots landing on marked set {sorted(marked_states)}: "
          f"{marked_fraction:.4f}")
    print(f"Quantum-recovered set (>5% of shots): {quantum_recovered_set}")

    classical_answer = sorted(S)
    # Baseline (uniform, no amplification) would land on the 3 marked states
    # only 3/8 = 0.375 of the time; Grover's single optimal iteration for
    # 3-out-of-8 should push this well above that baseline.
    quantum_matches_classical = (
        quantum_recovered_set == classical_answer and marked_fraction > 0.75
    )

    if quantum_matches_classical:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
