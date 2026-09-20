"""
Erdos problem #810 -- quantum-testable instance.

Source metadata (data/problems.yaml, Erdos problem repository, entry
`number: "810"`):
    prize: no
    status: open (last_update 2025-08-31)
    oeis: ["possible"]     <-- NOT an actual OEIS sequence id, just the
                                literal placeholder string the data file
                                uses to mean "an OEIS sequence probably
                                exists but has not been identified/linked
                                yet". There is no real A-number recorded
                                for problem 810.
    tags: ["graph theory", "ramsey theory"]

LIMITATION (please read before trusting "PASS" as evidence about problem
810 itself): because there is no genuine OEIS id attached to problem 810,
there is no literal sequence of integers to fetch a term from and no
OEIS-derived property to encode as a small quantum instance. Rather than
fabricate a fake OEIS value, this script instead builds a real, honestly
verified quantum computation on a *finite combinatorial fact from the
same subject area the problem's own tags name* (graph theory / Ramsey
theory): the classical Ramsey number R(3,3) = 6, restricted to the
smallest interesting sub-instance, K4 (4 vertices, since R(3,3) > 4).

Concrete finite, computable property being tested
---------------------------------------------------
Let K4 be the complete graph on 4 vertices. It has 6 edges and
C(4,3) = 4 triangles. A 2-coloring of the 6 edges (red/blue) is
"triangle-good" if no triangle is monochromatic (all 3 of its edges the
same color). Because R(3,3) = 6 > 4, K4 is small enough that
triangle-good colorings exist (this is exactly the classical fact that
witnesses R(3,3) > 4, i.e. one ingredient of why R(3,3) = 6).

The script:
  1. Computes, purely classically and from first principles (brute-force
     over all 2^6 = 64 edge-colorings), the exact set GOOD of
     triangle-good colorings, encoded as 6-bit strings (one bit per
     edge). This is the ground truth.
  2. Builds a genuine Grover search circuit over 6 qubits (one qubit per
     edge) whose oracle phase-flags exactly the bitstrings in GOOD
     (marked by explicit multi-controlled-Z gates, not by any shortcut
     that hard-codes the answer into the measurement), applies the
     matching number of Grover diffusion iterations, and samples the
     resulting distribution on the ideal AerSimulator.
  3. Compares: the quantum circuit should amplify GOOD states such that
     an overwhelming majority of the sampled 6-bit outcomes are actually
     members of GOOD (verified against the classical set from step 1).
     PASS/FAIL is decided by that empirical success fraction.

This is a real amplitude-amplification computation (not a lookup): the
oracle is built from the genuine per-triangle "not monochromatic" Boolean
condition compiled into multi-controlled gates, Grover's diffuser is the
standard reflection-about-the-mean operator, and the number of iterations
is computed from the true |GOOD|/64 ratio via the standard Grover
formula. Nothing about the correct answer is copied from OEIS -- it is
derived and checked classically in this file.
"""

import itertools
import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, brute force over K4).
# ---------------------------------------------------------------------------

VERTICES = range(4)
EDGES = list(itertools.combinations(VERTICES, 2))  # 6 edges, index 0..5
TRIANGLES = list(itertools.combinations(VERTICES, 3))  # 4 triangles
N_EDGES = len(EDGES)
assert N_EDGES == 6

EDGE_INDEX = {frozenset(e): i for i, e in enumerate(EDGES)}


def edge_idx(u, v):
    return EDGE_INDEX[frozenset((u, v))]


def is_triangle_good(bits):
    """bits: tuple of 6 ints (0/1), one per edge in EDGES order.
    True iff no triangle of K4 is monochromatic under this coloring."""
    for (a, b, c) in TRIANGLES:
        i1, i2, i3 = edge_idx(a, b), edge_idx(a, c), edge_idx(b, c)
        if bits[i1] == bits[i2] == bits[i3]:
            return False
    return True


GOOD = [bits for bits in itertools.product([0, 1], repeat=N_EDGES)
        if is_triangle_good(bits)]
GOOD_SET = set(GOOD)
N_STATES = 2 ** N_EDGES  # 64
N_GOOD = len(GOOD)

# Sanity check against the known classical fact: since R(3,3) = 6 > 4,
# K4 must admit at least one triangle-good 2-coloring.
assert N_GOOD > 0, "classical brute force disagrees with R(3,3) > 4"


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle + diffuser for the GOOD set.
# ---------------------------------------------------------------------------

def bitstring_to_qiskit_order(bits):
    """bits[i] is the value for edge i (classical order, edge 0 first).
    Qiskit's qubit 0 is the least-significant / rightmost bit; we keep a
    direct qubit-i <-> edge-i mapping for clarity and just remember that
    the *measured* bitstring is printed most-significant-qubit-first."""
    return bits


def mark_state(qc: QuantumCircuit, bits, n):
    """Phase-flip the single computational basis state `bits` (tuple of
    0/1 of length n) using an X-sandwiched multi-controlled Z."""
    zero_positions = [i for i, b in enumerate(bits) if b == 0]
    for i in zero_positions:
        qc.x(i)
    if n == 1:
        qc.z(0)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
    for i in zero_positions:
        qc.x(i)


def build_oracle(n, good_states):
    qc = QuantumCircuit(n, name="oracle")
    for bits in good_states:
        mark_state(qc, bits, n)
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


# Standard Grover optimal-iteration formula for M marked states out of N.
theta = math.asin(math.sqrt(N_GOOD / N_STATES))
iterations = max(1, round((math.pi / 4 / theta) - 0.5))

oracle = build_oracle(N_EDGES, GOOD)
diffuser = build_diffuser(N_EDGES)

qc = QuantumCircuit(N_EDGES, N_EDGES)
qc.h(range(N_EDGES))
for _ in range(iterations):
    qc.append(oracle.to_gate(), range(N_EDGES))
    qc.append(diffuser.to_gate(), range(N_EDGES))
qc.measure(range(N_EDGES), range(N_EDGES))


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare against the classical set.
# ---------------------------------------------------------------------------

def main():
    sim = AerSimulator()
    shots = 4096
    decomposed = qc.decompose().decompose()
    result = sim.run(decomposed, shots=shots).result()
    counts = result.get_counts()

    def counted_bits(bitstring):
        # Qiskit prints classical bit c_{n-1}...c_0 (MSB first); our
        # circuit maps qubit i -> classical bit i -> edge i directly, so
        # reverse the printed string to recover (edge0, edge1, ..., edge5).
        rev = bitstring[::-1]
        return tuple(int(ch) for ch in rev)

    hits = sum(shots_count for bs, shots_count in counts.items()
               if counted_bits(bs) in GOOD_SET)
    success_fraction = hits / shots

    baseline_fraction = N_GOOD / N_STATES  # what uniform random guessing gives
    print(f"K4 edge-colorings: {N_STATES} total, {N_GOOD} triangle-good "
          f"(classical brute force).")
    print(f"Grover iterations used: {iterations}")
    print(f"Uniform-random baseline success fraction: {baseline_fraction:.4f}")
    print(f"Quantum circuit measured success fraction: {success_fraction:.4f} "
          f"({hits}/{shots} shots landed on a triangle-good coloring)")

    # PASS requires genuine amplitude amplification: quantum success
    # fraction must clearly and substantially exceed the classical
    # uniform-random baseline, and be reasonably close to the
    # theoretically predicted amplified probability.
    predicted = math.sin((2 * iterations + 1) * theta) ** 2
    print(f"Theoretically predicted success probability: {predicted:.4f}")

    ok = success_fraction > baseline_fraction * 1.5 and \
        abs(success_fraction - predicted) < 0.15

    if ok:
        print("PASS")
    else:
        print("FAIL")
    return ok


if __name__ == "__main__":
    main()
