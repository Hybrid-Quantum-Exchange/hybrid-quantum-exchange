"""
Erdos problem #337 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry "number: 337"):
    prize: no
    status: disproved (Lean), last update 2025-12-10
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics", "additive basis"]

LIMITATION, stated honestly up front: problem #337 carries no OEIS sequence id
("N/A" in the source data), so there is no literal OEIS term to look up or to
have a quantum circuit reproduce. What the entry does give us is a real,
well-defined mathematical topic from its tags -- "additive basis" / "additive
combinatorics" -- so this script builds a genuine finite/computable property
from that topic rather than fabricating a fake OEIS lookup: it is NOT a
verification of problem #337 itself (that would require the actual Lean
disproof), it is a small additive-basis search problem in the same subject
area, chosen because it is something a small Grover circuit can honestly
compute.

Classical property being tested (computed from first principles below, no
external data):
    Let n = 6, working in Z_n = {0, 1, ..., 5}. A subset S of Z_n is an
    "additive basis of order 2" for Z_n if the sumset
        S + S = { (a + b) mod n : a, b in S }
    equals all of Z_n. Every subset S of Z_n is encoded as an n-bit string
    x in {0,1}^n (bit i = 1 iff i is in S). The search space therefore has
    2^n = 64 candidate subsets -- a small, fully enumerable instance.

    property(x): the subset S encoded by bitstring x is a nonempty additive
    basis of order 2 for Z_6.

The script first solves this by brute-force classical enumeration over all 64
bitstrings (ground truth). It then builds a Grover search circuit over 6
qubits whose oracle is the exact diagonal phase flip on the classically
precomputed set of marked (basis) bitstrings, runs it on the ideal
AerSimulator, and checks that the state(s) Grover amplifies are indeed the
correct classically-marked additive bases. PASS/FAIL is a genuine agreement
check between the quantum measurement outcome and the classical answer, not a
restated constant.
"""

from __future__ import annotations

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCMTGate, ZGate, DiagonalGate
from qiskit_aer import AerSimulator

N = 6  # size of Z_n; also number of qubits (subset membership bits)


def sumset_covers_zn(bits: tuple[int, ...], n: int) -> bool:
    """True iff the subset encoded by `bits` (bit i == 1 means i in S) is
    nonempty and S + S (mod n) covers all of Z_n."""
    s = [i for i, b in enumerate(bits) if b == 1]
    if not s:
        return False
    covered = set()
    for a in s:
        for b in s:
            covered.add((a + b) % n)
    return len(covered) == n


def classical_ground_truth(n: int) -> list[int]:
    """Brute-force, from first principles: enumerate all 2^n bitstrings,
    return the list of integer indices whose subset is an additive basis of
    order 2 for Z_n."""
    marked = []
    for x in range(2 ** n):
        bits = tuple((x >> i) & 1 for i in range(n))
        if sumset_covers_zn(bits, n):
            marked.append(x)
    return marked


def build_oracle(n: int, marked_indices: list[int]) -> QuantumCircuit:
    """Phase oracle: applies -1 to exactly the computational basis states
    listed in marked_indices, identity to every other state. Built as an
    explicit diagonal unitary so its correctness is checkable by inspection,
    not by reusing whatever produced marked_indices."""
    diag = np.ones(2 ** n, dtype=complex)
    for idx in marked_indices:
        diag[idx] = -1.0
    qc = QuantumCircuit(n, name="oracle")
    qc.append(DiagonalGate(diag.tolist()), list(range(n)))
    return qc


def build_diffuser(n: int) -> QuantumCircuit:
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    if n == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), n - 1, 1)
        qc.append(mcz, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover(n: int, marked_indices: list[int], shots: int = 4096) -> dict:
    m = len(marked_indices)
    total = 2 ** n
    # Standard optimal Grover iteration count for m marked out of total.
    iterations = max(1, round((math.pi / 4) * math.sqrt(total / m)))

    oracle = build_oracle(n, marked_indices)
    diffuser = build_diffuser(n)

    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n))
        qc.append(diffuser.to_instruction(), range(n))
    qc.measure(range(n), range(n))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main() -> None:
    marked = classical_ground_truth(N)
    print(f"Z_{N}: classical brute-force search over {2 ** N} subsets")
    print(f"Additive bases of order 2 for Z_{N} found classically: {len(marked)}")
    for idx in marked:
        s = [i for i in range(N) if (idx >> i) & 1]
        print(f"  index {idx:2d} (binary {idx:0{N}b}) -> subset {s}")

    assert len(marked) > 0, "expected at least one additive basis to exist for Z_6"

    counts = run_grover(N, marked)
    # Qiskit bitstrings are printed MSB..LSB over the classical register,
    # with register bit 0 as the rightmost character.
    total_shots = sum(counts.values())
    hits_on_marked = 0
    top_outcomes = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    print("\nTop Grover measurement outcomes (bitstring: count):")
    for bitstring, count in top_outcomes:
        idx = int(bitstring[::-1], 2)  # undo qiskit's bit ordering
        is_marked = idx in marked
        if is_marked:
            hits_on_marked += count
        print(f"  {bitstring} -> index {idx:2d} marked={is_marked} count={count}")

    # Total probability mass Grover placed on classically-verified marked states.
    marked_mass = 0
    for bitstring, count in counts.items():
        idx = int(bitstring[::-1], 2)
        if idx in marked:
            marked_mass += count
    marked_fraction = marked_mass / total_shots

    most_likely_bitstring, most_likely_count = top_outcomes[0]
    most_likely_idx = int(most_likely_bitstring[::-1], 2)
    quantum_answer_is_valid_basis = most_likely_idx in marked

    print(f"\nFraction of shots landing on a classically-verified additive basis: "
          f"{marked_fraction:.3f}")
    print(f"Most likely measured outcome corresponds to index {most_likely_idx}, "
          f"which is a valid additive basis of order 2 for Z_{N}: "
          f"{quantum_answer_is_valid_basis}")

    # Verification criterion: Grover's most likely outcome must itself be a
    # classically-verified additive basis, AND Grover must have concentrated
    # a clear majority of shots on the marked subspace (well above the
    # uniform-random baseline len(marked)/2^N).
    baseline = len(marked) / (2 ** N)
    verified = quantum_answer_is_valid_basis and marked_fraction > max(0.5, 1.5 * baseline)

    print(f"Uniform-random baseline probability of hitting a marked state: {baseline:.4f}")
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
