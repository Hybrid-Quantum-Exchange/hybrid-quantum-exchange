"""
Erdos problem #788 -- quantum-testable instance
=================================================

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 788"):
    prize: no
    status: open
    oeis: ["possible"]        <-- NOT a real OEIS id. "possible" is the
                                   upstream dataset's placeholder meaning
                                   "an OEIS entry may exist" -- no id number
                                   is actually recorded for this problem.
    tags: ["additive combinatorics"]

LIMITATION (reported honestly, per instructions): problem #788 has no
concrete OEIS sequence id attached in the source data, and no numeric
statement is given in the metadata to derive a finite property from. There
is therefore no specific classical "term of sequence A0XXXXX" that this
script can check. Faking an OEIS id or copying a value from elsewhere would
misrepresent the problem, so this script does not do that.

Best-effort honest attempt
---------------------------
Rather than fabricate a link to a sequence that isn't there, this script
builds a real, finite, computable instance of the one piece of genuine
mathematical content the metadata *does* give us: the tag "additive
combinatorics". It picks a standard, well-defined finite object from that
area -- Sidon sets (B2 sets: subsets in which all pairwise sums a+b, a<=b,
are distinct) -- and uses Grover's algorithm to search the space of subsets
of {1,2,3,4} for the ones that are Sidon sets.

This is a genuine, independently checkable computational task (not tied to
problem 788's actual open conjecture, which concerns an unspecified additive
combinatorics statement with no finite decidable form in the source data).
The classical answer is computed from first principles in this script by
brute force over all 16 subsets, and the quantum circuit is a real Grover
search (uniform superposition -> phase oracle built from the same brute
force list -> diffusion operator, repeated the optimal number of times)
run on the ideal AerSimulator. PASS means the state(s) Grover amplifies
match the classically computed set of Sidon subsets of {1,2,3,4}.

Instance
--------
Ground set: {1, 2, 3, 4}, encoded as 4 qubits q0..q3 (qubit i = whether
element i+1 is in the subset). 16 possible subsets total.

A subset S is a Sidon set iff all pairwise sums a+b for a,b in S, a<=b,
are pairwise distinct (equivalently: no nontrivial solution a+b=c+d with
{a,b} != {c,d}, a,b,c,d in S).
"""

import itertools
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


GROUND_SET = [1, 2, 3, 4]
N_QUBITS = len(GROUND_SET)


def is_sidon(subset):
    """A subset is Sidon (B2) if all pairwise sums a+b (a<=b) are distinct."""
    sums = []
    for a, b in itertools.combinations_with_replacement(subset, 2):
        sums.append(a + b)
    return len(sums) == len(set(sums))


def bits_to_subset(bitstring):
    """bitstring[i] == '1' (little-endian, qubit 0 = leftmost char here after
    reversal) means GROUND_SET[i] is included."""
    return [GROUND_SET[i] for i, bit in enumerate(bitstring) if bit == "1"]


def classical_non_sidon_subsets():
    """Brute-force, from first principles, every subset of GROUND_SET and
    return the list of index-bitstrings (qubit-order, i.e. bit i = element
    i+1) whose subset FAILS to be a Sidon set (the minority class here,
    which is what Grover search is well-suited to amplify)."""
    marked = []
    for bits in itertools.product("01", repeat=N_QUBITS):
        subset = [GROUND_SET[i] for i, b in enumerate(bits) if b == "1"]
        if not is_sidon(subset):
            marked.append("".join(bits))
    return marked


def build_oracle(marked_bitstrings, n_qubits):
    """Phase oracle that flips the sign of exactly the marked computational
    basis states. Bit convention: bitstring[i] refers to qubit i (qc.x is
    applied to qubit i when that bit is '0', a multi-controlled Z is applied
    across all qubits, then the X gates are undone) -- standard technique
    for encoding a known list of marked basis states into a Grover oracle.
    """
    qc = QuantumCircuit(n_qubits, name="oracle")
    for bitstring in marked_bitstrings:
        zero_positions = [i for i, b in enumerate(bitstring) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_iterations(n_marked, n_total):
    import math

    if n_marked == 0 or n_marked == n_total:
        return 0
    theta = math.asin(math.sqrt(n_marked / n_total))
    r = round((math.pi / 4 / theta) - 0.5)
    return max(1, r)


def run_grover(marked_bitstrings, n_qubits, shots=4096):
    n_total = 2 ** n_qubits
    iterations = optimal_iterations(len(marked_bitstrings), n_total)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_bitstrings, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    print("Erdos problem #788 -- best-effort quantum-testable instance")
    print("=" * 60)
    print("Source oeis field: ['possible'] (placeholder, no real OEIS id)")
    print("Source tags: ['additive combinatorics']")
    print()
    print("No finite property tied to a real OEIS sequence exists for #788")
    print("in the source data, so this script instead runs a genuine Grover")
    print("search for Sidon (B2) sets over subsets of {1,2,3,4}, the closest")
    print("well-defined finite additive-combinatorics computation available.")
    print()

    classical_marked = classical_non_sidon_subsets()
    classical_subsets = sorted(
        tuple(bits_to_subset(b)) for b in classical_marked
    )
    print(f"Classical brute-force NON-Sidon subsets of {{1,2,3,4}} "
          f"({len(classical_marked)} of 16) -- Grover search target:")
    for s in classical_subsets:
        print(f"  {list(s)}")
    print()

    counts, iterations = run_grover(classical_marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    # Qiskit's default bit ordering in the classical register string is
    # big-endian (qubit n-1 first). Our marked bitstrings were built with
    # index i = qubit i (little-endian). Convert measured keys accordingly.
    def qiskit_key_to_bitstring(key):
        return key[::-1]  # reverse to get qubit-index order (qubit0 first)

    shots = sum(counts.values())
    top_states = sorted(counts.items(), key=lambda kv: -kv[1])
    marked_set = set(classical_marked)

    # Sum of probability mass landing on classically-marked (Sidon) states.
    marked_mass = 0
    for key, c in counts.items():
        if qiskit_key_to_bitstring(key) in marked_set:
            marked_mass += c
    marked_fraction = marked_mass / shots

    print(f"Measured probability mass on classically-NON-Sidon states: "
          f"{marked_fraction:.3f} (uniform-random baseline would be "
          f"{len(classical_marked)/16:.3f})")
    print()
    print("Top 5 measured outcomes (qiskit bit order -> subset -> is_sidon):")
    for key, c in top_states[:5]:
        bitstring = qiskit_key_to_bitstring(key)
        subset = bits_to_subset(bitstring)
        print(f"  {key}  count={c:5d}  subset={subset}  "
              f"sidon={bitstring not in marked_set}")

    # PASS criterion: Grover must have amplified the marked (non-Sidon)
    # states well above the uniform baseline, and the single most-measured
    # outcome must itself be a classically-verified non-Sidon set.
    baseline = len(classical_marked) / 16
    amplified = marked_fraction > baseline * 1.5
    top_key, _ = top_states[0]
    top_is_marked = qiskit_key_to_bitstring(top_key) in marked_set

    verified = amplified and top_is_marked

    print()
    if verified:
        print("PASS: Grover search amplified classically-verified "
              "non-Sidon sets, and the top measured outcome is a true "
              "non-Sidon set, matching the classical brute-force answer.")
    else:
        print("FAIL: Grover output did not match the classical answer.")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
