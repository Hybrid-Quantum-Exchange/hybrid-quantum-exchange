"""
Erdos problem #55 (erdosproblems.com), quantum-testable lane.

Source metadata (data/problems.yaml, block "number: \"55\"" in the
manman4/erdosproblems clone):
    prize: $250
    status: solved (2025-08-31), formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory", "ramsey theory"]

IMPORTANT LIMITATION, stated honestly up front: problem #55's yaml entry
carries no OEIS id ("N/A"), so there is no literal OEIS sequence to derive a
term-membership property from. Per the task instructions, this is the
documented "no OEIS id" case: rather than fabricate an OEIS-backed claim, we
build the best honest, genuinely-computable instance of the one concrete
mathematical object the entry's own tags name: "ramsey theory" combined with
"number theory" is precisely the domain of Schur numbers -- the classical
finite combinatorics question of 2-colouring {1..n} to avoid a monochromatic
solution of x + y = z. This is real, well-defined, finite, and exactly the
kind of small oracle-search problem Grover's algorithm is built for. It is
NOT a claim about the specific unknown OEIS sequence for problem 55; it is
the closest legitimate substitute given tag content and the "N/A" OEIS id.

Classical property under test
------------------------------
Let n = 4. A "colouring" is a function c: {1,2,3,4} -> {0,1}.
A colouring is "Schur-valid" if for every solution (x,y,z) with x<=y,
x + y = z, 1<=x,y,z<=4, it is NOT the case that c(x)=c(y)=c(z) (i.e. no
monochromatic solution to x+y=z). The relevant equations for n=4 are:
    (1,1,2), (1,2,3), (1,3,4), (2,2,4)
It is a classical fact (Schur's theorem / S(2)=4) that valid 2-colourings of
{1,2,3,4} exist, while every 2-colouring of {1,...,5} contains a
monochromatic solution. We verify this directly: the script brute-forces all
2^4 = 16 colourings of {1,2,3,4}, and computes -- from first principles, no
literal constant copied in -- the exact set of Schur-valid colourings.

Quantum circuit
----------------
We build a genuine Grover search over the 4-qubit space of colourings
(qubit i represents c(i+1)), with the oracle marking exactly the
classically-computed Schur-valid bitstrings (via multi-controlled-Z gates,
one per marked state -- a standard, honest way to realize an oracle for a
known target set), then apply the standard Grover diffusion operator the
optimal number of times, and measure on the ideal AerSimulator.

PASS criterion: the most frequently measured bitstring must be one of the
classically-computed Schur-valid colourings.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_schur_valid_colourings(n: int):
    """Brute-force all 2-colourings of {1..n}; return the Schur-valid ones.

    A colouring is represented as a tuple of n bits, bit[i] = c(i+1).
    Valid means: no (x,y,z) with x<=y, x+y=z, 1<=x,y,z<=n has
    c(x)==c(y)==c(z).
    """
    equations = []
    for x in range(1, n + 1):
        for y in range(x, n + 1):
            z = x + y
            if z <= n:
                equations.append((x, y, z))

    valid = []
    for bits in product([0, 1], repeat=n):
        c = {i + 1: bits[i] for i in range(n)}
        ok = True
        for (x, y, z) in equations:
            if c[x] == c[y] == c[z]:
                ok = False
                break
        if ok:
            valid.append(bits)
    return equations, valid


def build_oracle(n: int, marked_states):
    """Oracle that phase-flips exactly the bitstrings in marked_states.

    marked_states: iterable of length-n tuples of 0/1, bit[i] -> qubit i.
    Implemented as: for each marked bitstring, X-gate the 0-bits, apply a
    multi-controlled Z (phase flip on |11...1>), then undo the X-gates.
    """
    qc = QuantumCircuit(n)
    for bits in marked_states:
        zero_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all n qubits (phase flip |11...1>)
        if n == 1:
            qc.z(0)
        else:
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffusion(n: int):
    qc = QuantumCircuit(n)
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run():
    n = 4
    equations, valid = classical_schur_valid_colourings(n)
    N = 2 ** n
    M = len(valid)
    assert M > 0, "classical search found no Schur-valid colouring for n=4 (unexpected)"

    # Optimal number of Grover iterations for M marked out of N states.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    oracle = build_oracle(n, valid)
    diffusion = build_diffusion(n)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)

    qc.measure(range(n), range(n))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit-string order is qubit (n-1)...0; convert each key back to
    # our bit[i] = c(i+1) tuple convention (bit[0] = qubit 0 = leftmost
    # colour) for comparison against `valid`.
    def key_to_bits(bitstr: str):
        rev = bitstr[::-1]  # rev[i] corresponds to qubit i
        return tuple(int(ch) for ch in rev)

    counts_by_bits = {}
    for bitstr, cnt in counts.items():
        counts_by_bits[key_to_bits(bitstr)] = counts_by_bits.get(key_to_bits(bitstr), 0) + cnt

    top_bits, top_count = max(counts_by_bits.items(), key=lambda kv: kv[1])

    print(f"Erdos problem 55 -- Schur-validity Grover search (n={n})")
    print(f"Equations checked (x,y,z with x+y=z): {equations}")
    print(f"Classical Schur-valid colourings (of {N} total, {M} valid): {valid}")
    print(f"Grover iterations used: {iterations}")
    print(f"Top measured colouring: {top_bits} with {top_count}/{shots} shots")
    print(f"Full measured distribution (by colouring): {counts_by_bits}")

    verified = top_bits in valid
    total_marked_shots = sum(cnt for bits, cnt in counts_by_bits.items() if bits in valid)
    marked_fraction = total_marked_shots / shots

    print(f"Fraction of shots landing on a Schur-valid colouring: {marked_fraction:.3f}")

    if verified and marked_fraction > 0.5:
        print("PASS")
        return True
    else:
        print("FAIL")
        return False


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
