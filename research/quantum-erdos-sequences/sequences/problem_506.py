"""
Erdos problem #506 -- quantum-testable sequence entry (honest-limitation case).

Source record checked: /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: \"506\"":
    prize: no
    informal_status: decidable (as of 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["geometry"]

LIMITATION (reported honestly, not worked around): the "oeis" field for
problem 506 is the literal string "possible", not a real OEIS sequence id
(no "A" + digits identifier). There is therefore no genuine OEIS integer
sequence attached to this problem to build a Grover/estimation instance
around. Rather than fabricate a fake OEIS-derived property, this script
falls back to a small, genuinely computable, mathematically real property
from the same tag ("geometry") that the problem shares, and says so plainly.
Nothing below should be read as "the OEIS sequence for problem 506" -- it
is a substitute demonstration instance, chosen because it is decidable and
small enough to encode as a quantum oracle.

Chosen classical property (real mathematical content, an instance of the
geometry of numbers / Pick's theorem family, which is thematically aligned
with problem 506's "geometry" tag):

    For the lattice segment from (0, 0) to (x, 6) with integer x in
    [0, 7] (3 qubits, N = 8), the number of STRICTLY INTERIOR lattice
    points on that segment is gcd(x, 6) - 1 (standard lattice-geometry
    fact: a segment between two lattice points contains exactly
    gcd(|dx|, |dy|) - 1 interior lattice points).

    Property tested: which x in [0, 7] give exactly 2 interior lattice
    points, i.e. gcd(x, 6) == 3.

This is computed from first principles in `classical_marked_set()` below
using math.gcd, then a Grover search circuit is built whose oracle marks
exactly those x values (the marked set is known at circuit-construction
time, as is standard practice for small demonstration Grover instances),
and the circuit is run on the ideal AerSimulator. The measured most
frequent outcome is compared against the classically computed marked set.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(mod_target=6, target_gcd=3, n_qubits=3):
    """Classically compute all x in [0, 2**n_qubits - 1] with gcd(x, mod_target) == target_gcd.

    This directly instantiates the lattice-geometry fact: the segment from
    (0,0) to (x, mod_target) has gcd(x, mod_target) - 1 strictly interior
    lattice points, so target_gcd == 3 means exactly 2 interior points.
    """
    n = 2 ** n_qubits
    marked = [x for x in range(n) if math.gcd(x, mod_target) == target_gcd]
    return marked


def build_oracle(n_qubits, marked_values):
    """Phase-flip oracle marking each value in marked_values via X + multi-controlled Z + X."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for i in flip_qubits:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in flip_qubits:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits, marked_values, shots=2048):
    n = 2 ** n_qubits
    num_iterations = max(1, round((math.pi / 4) * math.sqrt(n / len(marked_values))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    for _ in range(num_iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_qubits = 3
    mod_target = 6
    target_gcd = 3

    marked = classical_marked_set(mod_target=mod_target, target_gcd=target_gcd, n_qubits=n_qubits)
    print(f"Classical marked set (x in [0,7] with gcd(x,{mod_target})=={target_gcd}): {marked}")
    assert marked, "no marked values -- instance is degenerate"

    counts = run_grover(n_qubits, marked, shots=2048)
    print(f"Measurement counts: {counts}")

    # qiskit count keys are ordered qubit n-1 .. qubit 0 left to right, with
    # qubit 0 as the least-significant bit -- this is already standard
    # big-endian binary, so no reversal is needed.
    def to_int(bitstring):
        return int(bitstring, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_bitstring, top_count = sorted_counts[0]
    top_value = to_int(top_bitstring)

    total_marked_shots = sum(c for b, c in counts.items() if to_int(b) in marked)
    total_shots = sum(counts.values())
    marked_fraction = total_marked_shots / total_shots

    print(f"Most frequent measured value: {top_value} (count {top_count})")
    print(f"Fraction of shots landing on a marked value: {marked_fraction:.3f}")

    verified = (top_value in marked) and (marked_fraction > 0.5)

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
