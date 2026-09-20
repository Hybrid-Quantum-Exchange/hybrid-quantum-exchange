"""
Erdos problem #523 -- quantum-testable companion script.

Erdos problem #523 (per erdosproblems.com / the manman4/erdosproblems data
export, data/problems.yaml, entry "number: '523'") is tagged
["analysis", "probability", "polynomials"], status "proved", and its `oeis`
field is `["N/A"]` -- the dataset records NO OEIS sequence id for this
problem. There is therefore no genuine integer sequence tied to problem 523
that this script can search or verify a term of; any claim to the contrary
would be fabricated, which the task instructions explicitly forbid.

LIMITATION (stated up front, honestly): this script does NOT test a term of
an Erdos-523 sequence, because no such OEIS sequence exists in the source
data. What follows is the best-effort fallback the task asked for when no
OEIS id is available: a small, genuinely computable finite decision problem
drawn from the *topic* of problem 523 (random/coefficient-restricted
polynomials and the probability of having only real roots -- the same
subject area as the "analysis / probability / polynomials" tags), solved
both classically and with a real Grover-search quantum circuit, so the
PASS/FAIL check still has real mathematical content and is not a copied
literal value.

Concrete finite instance:
  Consider all monic quadratics  p(x) = x^2 + b*x + c  with coefficients
  b, c independently ranging over {-2, -1, +1, +2} (4 values each, encoded
  as 2 qubits per coefficient -> 16 candidates total, a 4-qubit Grover
  search space). A candidate is marked "good" iff its discriminant
  d = b^2 - 4*c is a strictly positive perfect square, i.e. p(x) has two
  *distinct rational* real roots (not merely real, and not a repeated
  root) -- a clean, finite, decision property in the same subject area as
  problem 523's tags (polynomials / real roots / the probability that a
  randomly-chosen-coefficient polynomial has a given root structure).

  Classically: enumerate all 16 indices, decode (b, c) from the 4 index
  bits via the fixed value list [-2, -1, 1, 2], compute the discriminant
  of x^2 + b*x + c for each, and record which indices are "good" (marked
  states for Grover). This gives exactly 2 good indices out of 16 (a
  1-in-8 minority), the regime where Grover search gives real, checkable
  amplification.

  Quantum: build a genuine Grover search circuit over the 4-qubit index
  space whose oracle marks exactly the classically-computed good indices
  (the oracle is built directly from that classical truth table, i.e. it is
  a verified black-box oracle, not a re-derivation of the arithmetic inside
  the circuit -- the circuit's job is the *search*, matching the "Grover
  search ... whichever fits" guidance for this kind of finite decision
  problem), run it on AerSimulator, and check that the highest-probability
  measured index is one of the classically-good indices with probability
  well above the uniform 1/16 baseline.

PASS/FAIL: prints PASS iff the most frequent Grover measurement outcome is
in the classically-computed good set, and the observed good-state
probability is significantly above the uniform baseline (2/16 = 0.125).
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


NUM_QUBITS = 4
N = 2 ** NUM_QUBITS  # 16 indices


def bit(i, k):
    """k-th bit (0 = LSB) of integer i."""
    return (i >> k) & 1


COEFF_VALUES = [-2, -1, 1, 2]


def two_bit_value(index, lo_qubit):
    """Read a 2-bit field (qubits lo_qubit, lo_qubit+1) as an int in [0,4)."""
    return bit(index, lo_qubit) + 2 * bit(index, lo_qubit + 1)


def is_real_rooted(index):
    """
    Classical ground truth for one index in [0, 16).
    Qubits 0,1 encode b = COEFF_VALUES[n], n = two_bit_value(index, 0).
    Qubits 2,3 encode c = COEFF_VALUES[m], m = two_bit_value(index, 2).
    Polynomial: x^2 + b*x + c.
    "Good" iff discriminant b^2 - 4*c is a strictly positive perfect
    square (two distinct rational real roots).
    """
    b = COEFF_VALUES[two_bit_value(index, 0)]
    c = COEFF_VALUES[two_bit_value(index, 2)]
    discriminant = b * b - 4 * c
    if discriminant <= 0:
        return False
    root = int(np.sqrt(discriminant))
    return root * root == discriminant


def classical_good_indices():
    return [i for i in range(N) if is_real_rooted(i)]


def build_oracle(good_indices):
    """Phase-flip oracle marking exactly `good_indices` among 3 qubits."""
    qc = QuantumCircuit(NUM_QUBITS, name="oracle")
    for idx in good_indices:
        bits = [bit(idx, k) for k in range(NUM_QUBITS)]
        # Flip qubits that are 0 in this index so a multi-controlled Z
        # fires exactly when the register equals `idx`.
        for k, b in enumerate(bits):
            if b == 0:
                qc.x(k)
        if NUM_QUBITS == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), NUM_QUBITS - 1, 1)
            qc.append(mcz, list(range(NUM_QUBITS)))
        for k, b in enumerate(bits):
            if b == 0:
                qc.x(k)
    return qc


def build_diffuser():
    qc = QuantumCircuit(NUM_QUBITS, name="diffuser")
    qc.h(range(NUM_QUBITS))
    qc.x(range(NUM_QUBITS))
    if NUM_QUBITS == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), NUM_QUBITS - 1, 1)
        qc.append(mcz, list(range(NUM_QUBITS)))
    qc.x(range(NUM_QUBITS))
    qc.h(range(NUM_QUBITS))
    return qc


def grover_iterations(num_good, num_total):
    theta = np.arcsin(np.sqrt(num_good / num_total))
    r = max(1, round((np.pi / (4 * theta)) - 0.5))
    return r


def run():
    good = classical_good_indices()
    assert len(good) > 0 and len(good) < N, "degenerate search space"

    oracle = build_oracle(good)
    diffuser = build_diffuser()
    iterations = grover_iterations(len(good), N)

    qc = QuantumCircuit(NUM_QUBITS, NUM_QUBITS)
    qc.h(range(NUM_QUBITS))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(NUM_QUBITS))
        qc.append(diffuser.to_gate(), range(NUM_QUBITS))
    qc.measure(range(NUM_QUBITS), range(NUM_QUBITS))

    backend = AerSimulator()
    shots = 4096
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit ordering: rightmost char in the bitstring is qubit 0.
    def outcome_to_index(bitstring):
        bits = bitstring[::-1]
        return sum(int(bits[k]) << k for k in range(NUM_QUBITS))

    index_counts = {}
    for bitstring, c in counts.items():
        idx = outcome_to_index(bitstring)
        index_counts[idx] = index_counts.get(idx, 0) + c

    best_index = max(index_counts, key=index_counts.get)
    good_prob = sum(index_counts.get(i, 0) for i in good) / shots
    baseline = len(good) / N

    print(f"Classical real-rooted indices (out of 0..{N-1}): {good}")
    print(f"Grover iterations used: {iterations}")
    print(f"Measured index counts: {index_counts}")
    print(f"Most frequent measured index: {best_index}")
    print(f"P(measure a real-rooted index) = {good_prob:.3f} "
          f"(uniform baseline = {baseline:.3f})")

    verified = (best_index in good) and (good_prob > baseline * 1.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    run()
