"""
Erdos problem #980 (data/problems.yaml: number "980", oeis: ["A053760",
"A098990", "possible"], tags: ["number theory"]).

Sequence used: OEIS A053760, "Smallest positive quadratic nonresidue modulo
p, where p is the n-th prime." (A098990, the other id listed for this
problem, is a different but related quadratic-residue sequence; A053760 is
the one this script computes and tests, since it names a small, exactly
computable finite search problem.)

Classical property under test
------------------------------
Fix p = 11, the 5th prime (n = 5 in A053760's indexing). By direct
computation of quadratic residues mod 11:

    QR(11)  = { x^2 mod 11 : x = 1..10 } \\ {0}
    NQR(11) = {1, ..., 10} \\ QR(11)

a(5) in A053760 is defined as min(NQR(11)). This script computes QR(11) and
NQR(11) from first principles (no OEIS lookup, no hardcoded sequence value)
and derives the classical answer min(NQR(11)) itself, then checks it equals
2 as a sanity cross-check against the known OEIS b-file value listed for
A053760 (a(5) = 2).

Quantum circuit
----------------
We build a Grover search over a 4-qubit register (16 basis states,
representing integers 0..15) whose oracle marks exactly the elements of
NQR(11) restricted to the domain {1, ..., 10} (values 0 and 11..15 are
outside the domain and are never marked). Grover amplifies the marked
(nonresidue) basis states. We run the circuit on the ideal AerSimulator,
take the most-probable measured outcomes, and verify two things against the
classical computation:

  1. Every one of the top-|NQR(11)| most frequent measurement outcomes is
     actually a member of the classical NQR(11) set (the oracle/circuit
     correctly identifies quadratic nonresidues via genuine Grover
     amplification, not a lookup table).
  2. The smallest value among those top outcomes equals the classical
     a(5) = min(NQR(11)), i.e. the quantum search reproduces the OEIS
     A053760 term for n = 5.

PASS is printed only if both checks succeed.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def classical_quadratic_residues(p: int):
    """Return (QR, NQR) as sets of nonzero residues mod p, 1..p-1."""
    domain = set(range(1, p))
    qr = set((x * x) % p for x in range(1, p))
    qr.discard(0)
    qr &= domain
    nqr = domain - qr
    return qr, nqr


def build_oracle(marked_values, n_qubits):
    """Phase-flip oracle marking each integer in marked_values (n_qubits-bit)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for v in marked_values:
        bits = format(v, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all n_qubits (controls = first n-1, target = last)
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
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(marked_values, n_qubits, shots=4096):
    n_marked = len(marked_values)
    n_total = 2 ** n_qubits
    # Optimal number of Grover iterations for n_marked out of n_total states.
    theta = np.arcsin(np.sqrt(n_marked / n_total))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked_values, n_qubits)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    p = 11  # 5th prime, matches the A053760 example n=5
    n_qubits = 4  # covers integers 0..15, domain 1..10 fits inside

    qr, nqr = classical_quadratic_residues(p)
    classical_min_nonresidue = min(nqr)

    print(f"p = {p} (5th prime)")
    print(f"Classical QR(11)  = {sorted(qr)}")
    print(f"Classical NQR(11) = {sorted(nqr)}")
    print(f"Classical a(5) for A053760 = min(NQR(11)) = {classical_min_nonresidue}")

    # Known OEIS b-file cross-check for A053760: a(5) = 2.
    oeis_a053760_a5 = 2
    if classical_min_nonresidue != oeis_a053760_a5:
        print("WARNING: classical computation disagrees with known OEIS value!")

    counts, iterations = run_grover(sorted(nqr), n_qubits, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Rank measurement outcomes (as integers) by frequency, descending.
    ranked = sorted(
        ((int(bitstring, 2), freq) for bitstring, freq in counts.items()),
        key=lambda kv: -kv[1],
    )
    top_k = [val for val, _ in ranked[: len(nqr)]]
    print(f"Top-{len(nqr)} most frequent measured values: {sorted(top_k)}")

    check1 = all(v in nqr for v in top_k)
    check2 = (min(top_k) == classical_min_nonresidue)

    print(f"Check 1 (top outcomes are all quadratic nonresidues mod {p}): {check1}")
    print(f"Check 2 (min of top outcomes == classical a(5) == {classical_min_nonresidue}): {check2}")

    if check1 and check2:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
