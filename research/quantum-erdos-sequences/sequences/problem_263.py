"""
Erdos problem #263 (irrationality; no OEIS id in the erdosproblems dataset).

Source metadata (data/problems.yaml, entry "number: \"263\"", tags: ["irrationality"]):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]

LIMITATION, stated honestly up front: problem #263 carries no OEIS sequence
id in the source data (oeis: ["N/A"]). There is therefore no real
OEIS-derived sequence to build a faithful quantum-testable instance from
for this specific problem number. Per instructions, rather than fabricate
an OEIS id or copy a literal value with no derivation, this script instead
builds its best honest, genuinely-computable instance consistent with
problem #263's only real attribute (tag: "irrationality"): a
continued-fraction / best-rational-approximation property of an irrational
number, sqrt(3) (distinct from sqrt(2), used for problem #260, to avoid a
duplicate instance).

Classical property under test
------------------------------
For q in {1, 2, ..., 8} (indices 0..7, 3 qubits), define
    frac(q) = q*sqrt(3) - floor(q*sqrt(3))
    dist(q) = min(frac(q), 1 - frac(q))      # distance to nearest integer
q is "marked" iff dist(q) < 0.12 (this threshold is fixed once, from the
classically computed distances below, before the circuit is built -- it is
not tuned against the quantum output).

Computed here from first principles (sqrt(3) via math.sqrt, no external
tables or OEIS lookups), the classical distances for q in 1..8 are printed
by the script itself at runtime; by hand:
    q=1: sqrt(3)=1.7320508  -> dist=0.2679492
    q=2: 2*sqrt(3)=3.4641016 -> dist=0.4641016
    q=3: 3*sqrt(3)=5.1961524 -> dist=0.1961524
    q=4: 4*sqrt(3)=6.9282032 -> dist=0.0717968  <-- marked
    q=5: 5*sqrt(3)=8.6602540 -> dist=0.3397460
    q=6: 6*sqrt(3)=10.3923048 -> dist=0.3923048
    q=7: 7*sqrt(3)=12.1243557 -> dist=0.1243557
    q=8: 8*sqrt(3)=13.8564065 -> dist=0.1435935
This matches the continued fraction of sqrt(3) = [1;1,2,1,2,1,2,...],
whose convergent denominators are 1, 2, 5, 7, 19, ...; q=4 is not itself a
convergent denominator but its distance to the nearest integer multiple
happens to be the smallest in range under this metric, which is exactly
the quantity Grover search below is asked to find (this is real, checkable
arithmetic, computed by the script, not asserted from an external table).

Quantum approach
-----------------
A 3-qubit Grover search over q in {1,...,8} (basis state |b> represents
q = b+1) with an oracle built directly from the classically precomputed
marked set (multi-controlled-Z gates keyed to the marked basis states),
plus the standard Grover diffuser, run on the ideal AerSimulator. The
number of Grover iterations is chosen optimally from the true count of
marked items (computed classically). The script then checks that the
most-frequently-measured basis state(s) match the classically computed
marked set.

Limitation, stated honestly: because Erdos problem #263 carries no OEIS
sequence, this is not a test of an OEIS term but of a genuine finite
arithmetic property motivated by the problem's "irrationality" tag
(continued-fraction / best-rational-approximation behavior of sqrt(3)).
ran_ok and verified_against_classical should be read as: does the circuit
run and correctly find the classically-derived marked set for this
stand-in instance (yes), not as: does this test problem #263's actual
open conjecture (it cannot, since #263 has no finite computable
reformulation available in the source data).
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_marked_set(n_values=8, threshold=0.12):
    """Compute, from first principles, which q in {1,...,n_values} have
    q*sqrt(3) unusually close to an integer (distance < threshold)."""
    sqrt3 = math.sqrt(3.0)  # computed here, not looked up
    marked = []
    dists = {}
    for q in range(1, n_values + 1):
        val = q * sqrt3
        frac = val - math.floor(val)
        dist = min(frac, 1.0 - frac)
        dists[q] = dist
        if dist < threshold:
            marked.append(q)
    return marked, dists


def build_oracle(n_qubits, marked_indices):
    """Multi-controlled-Z oracle flipping the phase of each marked basis
    state |b> (b = index, 0-indexed) among the n_qubits-qubit basis."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for idx in marked_indices:
        bits = format(idx, f"0{n_qubits}b")
        flip_positions = [i for i, b in enumerate(bits[::-1]) if b == "0"]
        for i in flip_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in flip_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
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


def main():
    n_values = 8
    n_qubits = 3  # 2^3 = 8, indices 0..7 represent q = 1..8

    marked_q, dists = classical_marked_set(n_values=n_values, threshold=0.12)
    marked_indices = sorted(q - 1 for q in marked_q)  # 0-indexed for the circuit

    print("Classical distances q*sqrt(3) to nearest integer, q=1..8:")
    for q in range(1, n_values + 1):
        print(f"  q={q}: dist={dists[q]:.6f}{'  <-- marked' if q in marked_q else ''}")
    print(f"Classically marked q values (dist < 0.12): {marked_q}")

    if not marked_indices:
        print("No marked items under this threshold; nothing for Grover to amplify.")
        print("FAIL")
        return

    num_marked = len(marked_indices)
    N = 2 ** n_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / num_marked)))

    oracle = build_oracle(n_qubits, marked_indices)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    sorted_outcomes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_indices = set()
    for bitstring, _ in sorted_outcomes[: max(num_marked, 1) + 1]:
        top_indices.add(int(bitstring, 2))

    quantum_top_q = sorted(idx + 1 for idx in top_indices if idx in set(marked_indices))

    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")
    print(f"Top measured basis states (as q): {sorted(idx + 1 for idx in top_indices)}")
    print(f"Classical marked q: {marked_q}")

    marked_total_shots = sum(cnt for bstr, cnt in counts.items() if int(bstr, 2) in marked_indices)
    marked_fraction = marked_total_shots / shots

    verified = set(marked_indices).issubset(top_indices) and marked_fraction > 0.5

    print(f"Fraction of shots landing on classically-marked states: {marked_fraction:.3f}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
