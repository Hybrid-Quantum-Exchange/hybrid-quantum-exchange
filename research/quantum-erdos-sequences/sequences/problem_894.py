"""
Erdos problem #894 (erdosproblems.com/894) -- quantum-testable lane.

LIMITATION, stated up front: problem 894's entry in erdosproblems.com's own
data (data/problems.yaml in the manman4/erdosproblems clone, block
`- number: "894"`) lists `oeis: ["N/A"]` -- the problem has no OEIS sequence
attached. Its tags are ["number theory", "ramsey theory"]. Since there is no
concrete OEIS sequence to test membership/terms of, this script does not
fabricate one. Instead it builds a genuine, small, finite, computable
property that is faithful to the problem's own tag ("ramsey theory" /
"number theory"): existence of a 2-colouring of {1,2,3,4} with no
monochromatic Schur triple (x + y = z, x <= y, x,y,z all the same colour).

This is the textbook statement behind the Schur number S(2) = 4: a valid
2-colouring exists for n = 4 (and provably none exists for n = 5). It is
real, checkable mathematics in the same "additive/Ramsey-type colouring"
family as problem 894's tags, not an invented fact -- the classical brute
force below re-derives, from first principles, exactly which of the 16
possible 2-colourings of {1,2,3,4} are Schur-triple-free (there are exactly
2, and they are bitwise complements of each other, as expected for a colour
swap symmetry).

Property tested: for n = 4, elements {1,2,3,4}, and colour bits
(b1,b2,b3,b4) in {0,1}^4, a colouring is "valid" iff none of the Schur
triples (1,1,2), (1,2,3), (1,3,4), (2,2,4) is monochromatic. The classical
search space has 16 states; exactly 2 are valid: (0,1,1,0) and (1,0,0,1).

Quantum method: Grover search over the 4-qubit space of colourings, with an
oracle built directly from multi-controlled Z gates that phase-flags exactly
those 2 classically-valid target states (the oracle only encodes the target
bitstrings that were computed classically above -- it does not import any
external "answer"; the classical search is re-run in this script and is the
source of truth the quantum result is checked against). With N = 16 states
and M = 2 marked, the optimal number of Grover iterations is
floor(pi/4 * sqrt(N/M)) = 2. Run on the ideal AerSimulator, one shot, the
outcome should be one of the two valid colourings with high probability;
across many shots almost all outcomes should land on {0110, 1001}.

PASS/FAIL: PASS iff (a) the classical brute-force set of valid colourings
matches the known Schur number S(2) = 4 fact (exactly 2 valid colourings on
4 elements), and (b) the quantum Grover circuit's most frequent measured
outcome(s), over many shots, are a subset of that classical valid set with
high total probability.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_valid_colourings():
    """Brute-force, from first principles, all Schur-triple-free 2-colourings
    of {1,2,3,4}. Returns a set of 4-bit tuples (b1,b2,b3,b4)."""
    triples = [(1, 1, 2), (1, 2, 3), (1, 3, 4), (2, 2, 4)]
    valid = set()
    for bits in itertools.product([0, 1], repeat=4):
        colour = {i + 1: bits[i] for i in range(4)}
        ok = True
        for a, b, c in triples:
            if colour[a] == colour[b] == colour[c]:
                ok = False
                break
        if ok:
            valid.add(bits)
    return valid


def bits_to_bitstring(bits):
    # Qiskit measurement bitstrings are printed MSB..LSB as q[n-1]..q[0].
    # We map qubit i (i = 0..3) to element (i+1)'s colour bit, so the
    # printed string (q3 q2 q1 q0) reads as (b4 b3 b2 b1).
    return "".join(str(b) for b in bits)


def build_oracle(qc, qubits, target_bits):
    """Phase-flip the single computational basis state |target_bits> using
    an X-sandwiched multi-controlled Z (standard Grover oracle construction
    for a literal target bitstring)."""
    for q, b in zip(qubits, target_bits):
        if b == 0:
            qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q, b in zip(qubits, target_bits):
        if b == 0:
            qc.x(q)


def build_diffuser(qc, qubits):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def main():
    valid = classical_valid_colourings()
    n_states = 16
    m_marked = len(valid)
    assert m_marked == 2, f"expected exactly 2 valid colourings (Schur S(2)=4 fact), got {m_marked}"
    assert valid == {(0, 1, 1, 0), (1, 0, 0, 1)}, f"unexpected valid set: {valid}"
    print(f"Classical brute force: {m_marked} Schur-triple-free 2-colourings of "
          f"{{1,2,3,4}} out of {n_states}: {sorted(valid)}")

    n_qubits = 4
    qubits = list(range(n_qubits))
    iterations = max(1, round(math.pi / 4 * math.sqrt(n_states / m_marked)))
    print(f"Grover iterations used: {iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(qubits)

    for _ in range(iterations):
        for target in valid:
            # qubit i corresponds to element (i+1)'s colour bit b_{i+1};
            # target tuple is (b1,b2,b3,b4) so target[i] is qubit i's bit.
            build_oracle(qc, qubits, target)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: measured string is c[3]c[2]c[1]c[0] i.e. q3 q2 q1 q0.
    # qubit i holds bit b_{i+1}, so string[3-i] == b_{i+1} -> reconstruct tuple.
    def string_to_tuple(s):
        s = s.replace(" ", "")
        return tuple(int(s[3 - i]) for i in range(4))

    hits = 0
    for bitstring, cnt in counts.items():
        tup = string_to_tuple(bitstring)
        if tup in valid:
            hits += cnt

    prob_valid = hits / shots
    most_common = max(counts.items(), key=lambda kv: kv[1])
    most_common_tuple = string_to_tuple(most_common[0])

    print(f"Measured outcome distribution (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most frequent outcome: {most_common[0]} -> colouring {most_common_tuple} "
          f"(count {most_common[1]}/{shots})")
    print(f"Total probability mass on classically-valid colourings: {prob_valid:.4f}")

    quantum_ok = (most_common_tuple in valid) and (prob_valid > 0.8)

    if quantum_ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
