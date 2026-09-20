"""
Erdos problem #645 (source: data/problems.yaml in the manman4/erdosproblems
clone) is listed as proved (Lean, 2025-11-23), with prize "no", and
oeis: ["N/A"] -- it has NO associated OEIS sequence id. Its tags are
["number theory", "additive combinatorics", "ramsey theory",
"arithmetic progressions"].

LIMITATION: because problem 645 carries no OEIS sequence, there is no
"sequence" to make quantum-testable in the sense the other lanes in this
library use. This script is the best honest substitute: it builds a real,
finite, computable decision problem that sits squarely inside the same
mathematical territory as problem 645's tags (Ramsey-type statements about
monochromatic arithmetic progressions), and verifies a genuine Grover-search
quantum circuit against the brute-force classical answer for a small
instance. It is NOT a claim that this recovers problem 645 itself, and
verified_against_classical below reflects a real circuit run, not a
literal OEIS lookup (there is no OEIS id to look up).

Classical property under test
------------------------------
Let N = 8, and 2-color the integers {0, ..., N-1} by
    color(x) = popcount(x) mod 2   (parity of number of 1-bits).
Consider all 3-term arithmetic progressions (a, a+d, a+2d) with
1 <= d and a, a+2d in range. For N = 8 there are exactly 12 such APs
(d=1: 6, d=2: 4, d=3: 2). Index them 0..11 (padded to a 4-qubit register,
indices 12..15 unused/invalid).

The property tested: "does a monochromatic 3-AP exist under this coloring,
and can Grover search find one of its indices?"

The classical answer (computed here from first principles, no OEIS lookup)
enumerates all 12 APs, evaluates color() on each triple, and records which
AP indices are monochromatic. That classical set of marked indices is then
used to build the Grover oracle explicitly (a standard, legitimate way to
build an oracle for a search problem whose marking predicate is known/
computable) -- the quantum circuit's job is the search over the 4-qubit
index register, not re-deriving the coloring.

Circuit
-------
A textbook Grover search over a 4-qubit index register (16 basis states,
12 "valid AP" states of which some are marked monochromatic), with the
optimal integer number of Grover iterations for the true marked-count,
run on the ideal AerSimulator (statevector, no noise).

Pass criterion
--------------
The most frequently measured basis state after Grover search must be one
of the classically-determined monochromatic-AP indices, AND we then
re-verify classically that this specific index's AP is indeed
monochromatic under color(). Both must hold for PASS.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def popcount_parity(x: int) -> int:
    return bin(x).count("1") % 2


def build_aps(n: int):
    """All 3-term APs (a, a+d, a+2d) within range(n), d >= 1."""
    aps = []
    d = 1
    while True:
        found_any = False
        for a in range(n):
            c = a + 2 * d
            if c < n:
                aps.append((a, a + d, c))
                found_any = True
        if not found_any:
            break
        d += 1
    return aps


def main():
    N = 8
    aps = build_aps(N)  # classically enumerated, first principles
    num_aps = len(aps)
    print(f"N = {N}, number of 3-term APs = {num_aps}")
    assert num_aps == 12, f"expected 12 APs for N=8, got {num_aps}"

    n_index_qubits = 4  # 2^4 = 16 >= 12
    num_states = 2 ** n_index_qubits

    # Classical ground truth: which AP indices are monochromatic.
    marked = []
    for idx, (a, b, c) in enumerate(aps):
        colors = {popcount_parity(a), popcount_parity(b), popcount_parity(c)}
        if len(colors) == 1:
            marked.append(idx)
    print(f"Classically monochromatic AP indices: {marked}")
    assert len(marked) > 0, "expected at least one monochromatic AP for N=8"

    for idx in marked:
        a, b, c = aps[idx]
        print(
            f"  AP #{idx}: ({a},{b},{c}) colors="
            f"({popcount_parity(a)},{popcount_parity(b)},{popcount_parity(c)})"
        )

    # ---- Build the Grover oracle from the classically-known marked set ----
    def apply_oracle(qc: QuantumCircuit, qubits, marked_indices, n_qubits):
        for m in marked_indices:
            bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian
            flip = [i for i, b in enumerate(bits) if b == "0"]
            for i in flip:
                qc.x(qubits[i])
            if n_qubits == 1:
                qc.z(qubits[0])
            else:
                qc.h(qubits[-1])
                qc.mcx(qubits[:-1], qubits[-1])
                qc.h(qubits[-1])
            for i in flip:
                qc.x(qubits[i])

    def apply_diffuser(qc: QuantumCircuit, qubits, n_qubits):
        for q in qubits:
            qc.h(q)
            qc.x(q)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
        for q in qubits:
            qc.x(q)
            qc.h(q)

    M = len(marked)
    theta = math.asin(math.sqrt(M / num_states))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations} (M={M}, N_states={num_states})")

    qc = QuantumCircuit(n_index_qubits, n_index_qubits)
    qubits = list(range(n_index_qubits))
    qc.h(qubits)
    for _ in range(iterations):
        apply_oracle(qc, qubits, marked, n_index_qubits)
        apply_diffuser(qc, qubits, n_index_qubits)
    qc.measure(qubits, qubits)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints classical bitstrings as c[n-1]...c[0]; with
    # qc.measure(qubits, qubits) that means qubit i sits at bit-weight 2^i
    # already, so the printed string, read directly as binary, equals the
    # little-endian index used when the oracle was built.
    int_counts = Counter()
    for bitstring, c in counts.items():
        idx = int(bitstring, 2)
        int_counts[idx] += c

    top_index, top_count = int_counts.most_common(1)[0]
    print(f"Most frequent measured index: {top_index} (count {top_count}/{shots})")

    quantum_found_marked = top_index in marked

    # Re-verify classically, independent of the marked list construction,
    # that the index Grover returned really is a monochromatic AP.
    verified = False
    if top_index < num_aps:
        a, b, c = aps[top_index]
        colors = {popcount_parity(a), popcount_parity(b), popcount_parity(c)}
        verified = len(colors) == 1

    ok = quantum_found_marked and verified
    print(f"quantum_found_marked={quantum_found_marked} reverified_classically={verified}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
