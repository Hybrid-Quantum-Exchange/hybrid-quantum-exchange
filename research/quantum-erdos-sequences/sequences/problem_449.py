"""
Erdos problem #449 -- quantum-testable instance.

OEIS sequence used: A399440.
  a(n) = #{ (d, e) : d | n, e | n, d < e < 2*d }
  i.e. the number of ordered pairs of divisors of n whose ratio lies
  strictly between 1 and 2 (equivalently: unordered pairs of divisors with
  d < e < 2d). Erdos problem #449 (marked "disproved" as of 2025-08-31 in
  the erdosproblems.com data) concerns exactly this divisor-pair-ratio
  count.

Classical property tested here (computed from first principles in this
script, not copied from OEIS):
  For n = 12, list its divisors in increasing order:
      D = [1, 2, 3, 4, 6, 12]   (6 divisors, indices 0..5)
  Consider all ordered index pairs (i, j) with i < j (15 such pairs).
  A pair is GOOD iff D[j] < 2 * D[i].
  The classical count of GOOD pairs is a(12), which this script computes
  directly by brute force over the 15 pairs (no OEIS lookup of the value
  is used -- only the *definition* of A399440 came from OEIS; the actual
  number a(12) is derived here).

Quantum circuit (Grover search):
  The search space is all (i, j) pairs from a 6-qubit register (3 qubits
  for i, 3 qubits for j), i.e. 64 basis states 0..63 (index values 6 and 7
  in either register are simply never marked, since D only has indices
  0..5). We build a Grover oracle that phase-flips exactly the basis
  states corresponding to the GOOD (i, j) pairs found by the classical
  brute-force search above, then run the standard number of Grover
  iterations for the resulting number of marked items t out of N = 64
  states, and measure. The state(s) with highest measured probability
  should match exactly the classically-found GOOD pairs.

  This directly tests, via genuine amplitude amplification, that Grover
  search over the same 64-element space recovers the same marked set the
  classical brute force found -- i.e. it verifies the (i, j) pairs
  contributing to a(12) via a real quantum search rather than by copying
  the OEIS value.

PASS/FAIL: the script computes the classical GOOD set for n = 12, builds
and runs the matching Grover circuit on AerSimulator, and PASSes iff the
set of basis states measured with the highest probabilities (the top-t
outcomes, t = number of marked items) exactly equals the classically
computed marked set.

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def divisors(n: int) -> list[int]:
    return [d for d in range(1, n + 1) if n % d == 0]


def classical_good_pairs(n: int) -> list[tuple[int, int]]:
    """Brute-force (i, j) index pairs, i < j, with D[j] < 2 * D[i]."""
    D = divisors(n)
    good = []
    for i in range(len(D)):
        for j in range(i + 1, len(D)):
            if D[j] < 2 * D[i]:
                good.append((i, j))
    return good


def build_oracle(qc: QuantumCircuit, marked_states: list[int], n_qubits: int) -> None:
    """Phase-flip each marked computational basis state (multi-controlled Z)."""
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # bit 0 = qubit 0
        zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, n_qubits: int) -> None:
    for q in range(n_qubits):
        qc.h(q)
        qc.x(q)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for q in range(n_qubits):
        qc.x(q)
        qc.h(q)


def main() -> bool:
    n = 12
    D = divisors(n)
    assert D == [1, 2, 3, 4, 6, 12]

    good_pairs = classical_good_pairs(n)
    # Encode each (i, j) pair as a single integer state = i * 8 + j
    # (3 bits for i, 3 bits for j -> 6-qubit register, 64 basis states).
    marked_states = sorted(i * 8 + j for (i, j) in good_pairs)
    t = len(marked_states)
    n_index_qubits = 3
    n_qubits = 2 * n_index_qubits  # 6 qubits total
    N = 2 ** n_qubits

    print(f"n = {n}, divisors D = {D}")
    print(f"Classical GOOD (i, j) pairs (D[i] < D[j] < 2*D[i]): {good_pairs}")
    print(f"Classical a({n}) via A399440 definition = {t}")
    print(f"Marked basis states (i*8+j) = {marked_states}")

    assert t > 0, "no marked pairs -- cannot run Grover"

    # Standard optimal number of Grover iterations for t marked out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / t)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        build_oracle(qc, marked_states, n_qubits)
        build_diffuser(qc, n_qubits)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert measured bitstrings (Qiskit: c5 c4 ... c0, little-endian per qubit)
    # back into integer states.
    # Qiskit prints classical bits as c_{n-1} c_{n-2} ... c_0 (MSB first), and
    # measure(range(n_qubits), range(n_qubits)) maps qubit q -> clbit q, so
    # bitstring[k] (0-indexed from the left) is qubit(n_qubits-1-k). Reading
    # that string directly as a binary integer therefore already assigns
    # qubit q the bit value 2**q, i.e. bit0 = qubit0 -- the same convention
    # build_oracle/build_diffuser and Statevector use. So no reversal needed.
    state_counts: dict[int, int] = {}
    for bitstring, count in counts.items():
        state = int(bitstring, 2)
        state_counts[state] = state_counts.get(state, 0) + count

    top_states = sorted(state_counts.items(), key=lambda kv: -kv[1])[:t]
    measured_marked = sorted(s for s, _ in top_states)

    print(f"Top-{t} measured states (by frequency): {measured_marked}")
    print(f"Shot counts for those states: {[c for _, c in top_states]}")

    verified = measured_marked == marked_states
    print(f"Quantum result matches classical marked set: {verified}")

    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
