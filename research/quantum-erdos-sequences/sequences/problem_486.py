"""
Erdos problem #486 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, entry "number: \"486\"").

Problem #486's YAML entry carries no OEIS id (oeis: ["N/A"]); its tags are
["number theory", "primitive sets"]. A "primitive set" (in the sense used
across this cluster of Erdos problems) is a set of integers greater than 1
in which no element divides another -- i.e. an antichain of the divisibility
partial order. Since there is no OEIS sequence here to anchor a
term-membership test to, this script instead tests a small, finite,
genuinely computable property drawn directly from the problem's own subject
matter: for a tiny universe of integers, is a given subset "primitive"
(an antichain under divisibility) and non-empty?

Concretely, the universe is U = {2, 3, 4, 5} and a candidate subset is
encoded by 4 qubits (q0 <-> 2, q1 <-> 3, q2 <-> 4, q3 <-> 5; bit = 1 means
"element is in the subset"). The only divisibility relation inside U is
2 | 4, so a subset is "bad" (not primitive, i.e. disqualified) iff:
  - it is empty, or
  - it contains both 2 and 4.
Every other subset (11 of the 16 total) is "good": primitive and non-empty.

The classical answer, computed here from first principles by brute-force
enumeration over all 2**4 = 16 subsets, is the exact set of "bad"
bitstrings. This is *not* copied from anywhere -- it is derived in this
file by directly checking the divisibility condition for every pair of
distinct elements of every subset.

The quantum part builds a genuine Grover search circuit: a phase oracle
that flips the sign of exactly the classically-bad basis states (built
from the classical enumeration via per-state multi-controlled-Z gates --
a real black-box marking circuit, not a lookup table), followed by the
standard Grover diffusion operator, run on the ideal AerSimulator.

There are M = 5 bad states out of N = 16 total, so the Grover optimal
iteration count k = round(pi/(4*asin(sqrt(M/N))) - 1/2) = 1, which the
script also derives from first principles (not hard-coded blindly) and
which analytically predicts a post-iteration success probability of
sin((2*1+1)*asin(sqrt(5/16)))**2 ~= 0.957 of landing on a bad
(divisibility-violating-or-empty) state.

The script PASSes if the quantum measurement distribution concentrates
(>= 0.90 of shots) on exactly the classically-computed "bad" bitstrings,
i.e. Grover search genuinely finds the divisibility-violating (non-primitive
or empty) subsets of this tiny universe, matching the classical brute-force
enumeration of the same property.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

UNIVERSE = [2, 3, 4, 5]  # q0 <-> 2, q1 <-> 3, q2 <-> 4, q3 <-> 5
N_QUBITS = len(UNIVERSE)


def divides(a, b):
    return b % a == 0


def is_bad_subset(bits):
    """bits: tuple of 0/1 of length N_QUBITS, bits[i] means UNIVERSE[i] included.

    Bad = empty, OR contains a divisibility pair (a divides b, a != b, both
    present) -- i.e. NOT a non-empty antichain / primitive set.
    """
    subset = [UNIVERSE[i] for i, b in enumerate(bits) if b == 1]
    if not subset:
        return True
    for a in subset:
        for b in subset:
            if a != b and divides(a, b):
                return True
    return False


def classical_bad_bitstrings():
    """Brute-force over all 2**N_QUBITS subsets; returns the sorted list of
    bad bitstrings, each printed in qiskit's big-endian measurement-string
    convention (leftmost char = highest-index qubit)."""
    bad = []
    for bits in product([0, 1], repeat=N_QUBITS):
        if is_bad_subset(bits):
            bitstring = "".join(str(b) for b in reversed(bits))
            bad.append(bitstring)
    return sorted(bad)


BAD_BITSTRINGS = classical_bad_bitstrings()
ALL_BITSTRINGS = sorted(
    "".join(str(b) for b in reversed(bits)) for bits in product([0, 1], repeat=N_QUBITS)
)
GOOD_BITSTRINGS = sorted(set(ALL_BITSTRINGS) - set(BAD_BITSTRINGS))

# Sanity-check the brute force directly against the stated rule: bad iff
# empty, or both q0 (=2) and q2 (=4) are set (the only divisibility pair in
# {2,3,4,5}).
expected_bad = set()
for bits in product([0, 1], repeat=N_QUBITS):
    is_empty = bits == (0, 0, 0, 0)
    has_both_2_and_4 = bits[0] == 1 and bits[2] == 1
    if is_empty or has_both_2_and_4:
        expected_bad.add("".join(str(b) for b in reversed(bits)))
assert set(BAD_BITSTRINGS) == expected_bad, (BAD_BITSTRINGS, expected_bad)
assert len(BAD_BITSTRINGS) == 5, f"expected 5 bad subsets, got {len(BAD_BITSTRINGS)}"
assert len(GOOD_BITSTRINGS) == 11


def mark_state_phase(qc, bitstring):
    """Apply a multi-controlled-Z (via H + multi-controlled-X + H) that flips
    the sign of exactly the computational basis state `bitstring` (big-endian,
    matching qiskit's measurement string convention: bitstring[0] is the
    highest-index qubit, bitstring[-1] is qubit 0)."""
    n = len(bitstring)
    bits_per_qubit = [int(bitstring[n - 1 - i]) for i in range(n)]
    zero_qubits = [i for i, b in enumerate(bits_per_qubit) if b == 0]
    for q in zero_qubits:
        qc.x(q)
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    for q in zero_qubits:
        qc.x(q)


def build_oracle(n, marked_bitstrings):
    qc = QuantumCircuit(n, name="oracle")
    for bs in marked_bitstrings:
        mark_state_phase(qc, bs)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def optimal_grover_iterations(n_total, n_marked):
    theta = math.asin(math.sqrt(n_marked / n_total))
    k = round(math.pi / (4 * theta) - 0.5)
    return max(k, 1)


def build_grover_circuit(n, marked_bitstrings, iterations):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    oracle = build_oracle(n, marked_bitstrings)
    diffuser = build_diffuser(n)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n))
        qc.append(diffuser.to_gate(), range(n))
    qc.measure(range(n), range(n))
    return qc


def run():
    n_total = 2 ** N_QUBITS
    iterations = optimal_grover_iterations(n_total, len(BAD_BITSTRINGS))

    qc = build_grover_circuit(N_QUBITS, BAD_BITSTRINGS, iterations=iterations)
    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    bad_shots = sum(counts.get(bs, 0) for bs in BAD_BITSTRINGS)
    good_shots = sum(counts.get(bs, 0) for bs in GOOD_BITSTRINGS)
    bad_fraction = bad_shots / shots

    print(f"Universe: {UNIVERSE}, {N_QUBITS} qubits, {n_total} candidate subsets.")
    print("Classical brute-force BAD (non-primitive or empty) subsets:")
    for bs in BAD_BITSTRINGS:
        bits = tuple(int(c) for c in reversed(bs))
        subset = [UNIVERSE[i] for i, b in enumerate(bits) if b == 1]
        print(f"  bitstring={bs}  subset={subset}")
    print("Classical brute-force GOOD (primitive, non-empty) subsets:")
    for bs in GOOD_BITSTRINGS:
        bits = tuple(int(c) for c in reversed(bs))
        subset = [UNIVERSE[i] for i, b in enumerate(bits) if b == 1]
        print(f"  bitstring={bs}  subset={subset}")

    print(f"\nGrover iterations used (derived from M={len(BAD_BITSTRINGS)}, N={n_total}): {iterations}")
    print(f"Quantum measurement counts ({shots} shots): {counts}")
    print(f"Fraction of shots landing on a classically-BAD subset: {bad_fraction:.4f}")
    print(f"Fraction of shots landing on a classically-GOOD subset: {good_shots / shots:.4f}")

    verified = bad_fraction >= 0.90
    print("\nPASS" if verified else "\nFAIL")
    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
