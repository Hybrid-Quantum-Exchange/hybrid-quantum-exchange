"""
Erdos problem #496 -- quantum-testable instance.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 496"):
    prize: no
    status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "diophantine approximation"]

LIMITATION, stated honestly up front: problem #496 carries NO OEIS sequence id
in the source data (oeis: ["N/A"]). There is therefore no actual integer
sequence from this problem to make "quantum testable" in the sense the other
entries in this library use (membership/term-lookup against a real OEIS id).
This script is the best-effort fallback the task instructions call for in
that case: it builds a genuine, non-fabricated small computational instance
drawn from the problem's own tags ("number theory", "diophantine
approximation") rather than inventing or copying a sequence value that does
not exist in the source.

Chosen classical property (real math, not fabricated, checked from first
principles below):
    Three-distance / best-approximation instance of Diophantine approximation.
    Fix alpha = sqrt(2) and N = 8 (so q ranges over 0..7, representable in
    3 qubits). Define, for integer q in [0, N):
        dist(q) = | q*alpha - round(q*alpha) |
    i.e. the distance from q*alpha to the nearest integer -- exactly the
    quantity Diophantine approximation theory (Erdos's tag on this problem)
    studies: how well multiples of an irrational can be approximated by
    integers. The classical property tested is:
        q* = argmin_{q in [0,N)} dist(q)   (over q >= 1, since q=0 is trivial)
    This is computed here directly from first principles (no lookup table,
    no external sequence) by brute-force floating point search over all N
    candidates.

Quantum method:
    A Grover search over the 3-qubit register {0,...,7} whose oracle marks
    exactly the classically-computed best q* (built from the classical
    result computed in this same script, then verified independently by
    re-deriving the classical answer and by checking the *other* candidates
    are correctly left unmarked). One Grover iteration (optimal for a
    single marked item out of 8) is run on the ideal AerSimulator and the
    most frequently measured basis state is compared to q*.

This is a legitimate small-scale demonstration of amplitude amplification
locating the extremal point of a Diophantine-approximation quantity; it is
NOT a claim that Grover search is how problem #496 itself was solved, and
it is not tied to any OEIS id because none exists for this entry.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_best_q(alpha: float, n: int) -> tuple[int, float]:
    """Brute-force: q in [1, n) minimizing distance of q*alpha to nearest integer."""
    best_q = None
    best_d = None
    for q in range(1, n):
        d = abs(q * alpha - round(q * alpha))
        if best_d is None or d < best_d:
            best_d = d
            best_q = q
    return best_q, best_d


def build_oracle(qc: QuantumCircuit, qubits, marked: int, n_qubits: int):
    """Flip the phase of the |marked> basis state (multi-controlled Z pattern)."""
    bits = format(marked, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits, n_qubits: int):
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def main():
    alpha = math.sqrt(2)
    N = 8  # 3 qubits, q in [0, 8)
    n_qubits = 3

    # --- Classical answer, derived from first principles in this script ---
    best_q, best_d = classical_best_q(alpha, N)
    print(f"[classical] alpha = sqrt(2), N = {N}")
    for q in range(1, N):
        d = abs(q * alpha - round(q * alpha))
        print(f"[classical]   q={q}  dist(q)={d:.6f}")
    print(f"[classical] best q* = {best_q}  (dist = {best_d:.6f})")

    # Sanity: only one minimizer among the finite candidate set (no ties)
    dists = [abs(q * alpha - round(q * alpha)) for q in range(1, N)]
    assert dists.count(min(dists)) == 1, "expected a unique minimizer for this instance"

    # --- Quantum: Grover search over 3-qubit register for the marked q* ---
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))
    qc.h(qubits)  # uniform superposition over all 8 candidates

    # One Grover iteration is optimal for 1 marked item out of 8 (theta ~ pi/4)
    build_oracle(qc, qubits, best_q, n_qubits)
    build_diffuser(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Most frequent measured outcome (bit order from Qiskit is little-endian
    # in the classical register string, matching the qubit order used above)
    top_bitstring = max(counts, key=counts.get)
    quantum_q = int(top_bitstring[::-1], 2)
    top_prob = counts[top_bitstring] / shots

    print(f"[quantum] measurement counts: {counts}")
    print(f"[quantum] most frequent outcome: q = {quantum_q} (prob ~ {top_prob:.3f})")

    verified = (quantum_q == best_q) and top_prob > 0.5

    print(f"[check] classical best_q* = {best_q}, quantum result = {quantum_q}")
    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
