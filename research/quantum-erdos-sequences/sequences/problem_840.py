"""
Erdos problem #840 -- quantum-testable lane (best-effort, limitation noted).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"840\"":
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["additive combinatorics", "sidon sets"]

LIMITATION (read before trusting the "OEIS id(s) used" claim): problem #840's
YAML record carries no OEIS sequence id -- the field is the literal string
"N/A". There is therefore no OEIS sequence to derive a finite term
membership/counting property from, as the assignment would otherwise prefer.
Rather than fabricate a fake OEIS id or bolt on an unrelated one, this script
honestly falls back to a small, genuinely finite, classically-checkable
property that is faithful to the problem's *tags* (additive combinatorics /
Sidon sets) instead of to a specific OEIS entry:

    A Sidon set (B2 set) is a set of non-negative integers S such that all
    pairwise sums a+b (a,b in S, a<=b) are distinct.

    Property tested: fix the base Sidon set S0 = {0, 1, 3} inside the
    universe U = {0, 1, ..., 7} (3 bits). Among the 5 remaining candidates
    U \\ S0 = {2, 4, 5, 6, 7}, find exactly which ones x satisfy
    "S0 union {x} is still a Sidon set" (i.e. adding x introduces no
    repeated pairwise sum). This is computed from first principles below
    by brute force over all 8 elements of U (candidates already in S0 are
    excluded/never marked) -- no OEIS value is copied or assumed.

This is a direct, finite combinatorial search problem (find the elements of
a small universe satisfying a boolean predicate -- "does adding this element
preserve the Sidon property"), which is exactly the shape Grover's algorithm
targets, so it is used here as a genuine quantum search circuit rather than
a trivial arithmetic wrapper.

Because problem #840 itself has no OEIS sequence, this script cannot claim
to "quantum-test" a specific OEIS sequence for #840; it demonstrates the
quantum search machinery on a small instance faithful to the problem's tags
(Sidon sets), and that limitation is reported honestly rather than papered
over.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


NUM_QUBITS = 3
UNIVERSE = list(range(2 ** NUM_QUBITS))  # {0,...,7}
BASE_SIDON_SET = {0, 1, 3}  # pairwise sums: 0,1,2,3,4,6 -- all distinct, genuine Sidon set


def is_sidon(elements):
    """True iff all pairwise sums a+b (a<=b, a,b in elements) are distinct."""
    sums = []
    elems = sorted(elements)
    for i, a in enumerate(elems):
        for b in elems[i:]:
            sums.append(a + b)
    return len(sums) == len(set(sums))


def classical_marked_values():
    """Brute force over the 3-bit universe {0,...,7}.

    Marked = x not already in BASE_SIDON_SET AND
             BASE_SIDON_SET union {x} is still a Sidon set.
    Computed from first principles (no OEIS lookup, no shortcut).
    """
    assert is_sidon(BASE_SIDON_SET), "base set must itself be a genuine Sidon set"
    marked = set()
    for x in UNIVERSE:
        if x in BASE_SIDON_SET:
            continue
        if is_sidon(BASE_SIDON_SET | {x}):
            marked.add(x)
    return marked


def build_oracle(marked_values, num_qubits):
    """Phase-flip oracle: |x> -> -|x> for x in marked_values, else |x>."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{num_qubits}b")[::-1]  # qubit i <-> bit i
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(marked_values, num_qubits, shots=4096):
    n_marked = len(marked_values)
    n_total = 2 ** num_qubits
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5)) if theta > 0 else 0

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = build_oracle(marked_values, num_qubits)
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(num_qubits))
        qc.append(diffuser.to_gate(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))
    qc = qc.decompose().decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    classical_marked = classical_marked_values()
    print(f"Base Sidon set S0 = {sorted(BASE_SIDON_SET)} (pairwise sums distinct: "
          f"{is_sidon(BASE_SIDON_SET)})")
    print(f"Classical brute-force result -- x in U\\S0 such that S0 U {{x}} is still "
          f"Sidon: {sorted(classical_marked)}")
    for x in UNIVERSE:
        if x in BASE_SIDON_SET:
            continue
        print(f"  x={x} ({x:03b}) -> S0 U {{x}} Sidon: {is_sidon(BASE_SIDON_SET | {x})}")

    counts, iterations = run_grover(classical_marked, NUM_QUBITS, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    quantum_marked_hits = {}
    for bitstring, count in counts.items():
        value = int(bitstring[::-1], 2)  # convert back to our little-endian int convention
        quantum_marked_hits[value] = quantum_marked_hits.get(value, 0) + count

    total_shots = sum(counts.values())
    marked_shots = sum(c for v, c in quantum_marked_hits.items() if v in classical_marked)
    marked_fraction = marked_shots / total_shots

    quantum_top_value = max(quantum_marked_hits, key=quantum_marked_hits.get)
    # Uniform-random baseline would land on a marked value with probability
    # len(classical_marked)/8. Grover amplification pushes this far above
    # baseline; a generous-but-still-discriminating threshold is used to
    # absorb finite-shot sampling noise.
    baseline = len(classical_marked) / 2 ** NUM_QUBITS
    verified = (quantum_top_value in classical_marked) and (marked_fraction > max(0.75, baseline + 0.25))

    print(f"Most frequent measured value: {quantum_top_value:03b} "
          f"(in classical marked set: {quantum_top_value in classical_marked})")
    print(f"Fraction of shots landing on a classically-marked value: {marked_fraction:.4f} "
          f"(uniform baseline would be {baseline:.4f})")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
