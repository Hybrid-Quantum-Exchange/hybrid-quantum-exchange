"""
Erdos problem #177 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, number: "177").

Problem #177's entry carries oeis: ["N/A"] -- there is no OEIS sequence
attached to it in the source data. Its tags are ["discrepancy",
"arithmetic progressions"], which places it in the same family as the
Erdos discrepancy problem: for a finite +/-1 sequence x(1..n), the
discrepancy of x is

    D(x) = max over integers d >= 1, k >= 1 with k*d <= n
           of | sum_{i=1..k} x(i*d) |

(the largest partial sum along any homogeneous arithmetic progression
i*d, 2d, 3d, ...). Because #177 itself has no attached OEIS id and no
literal numeric answer to check a quantum circuit against, this script
does NOT claim to test #177's OEIS sequence (there isn't one). Instead,
honestly scoped to what is actually computable and quantum-testable, it
builds a genuine small instance of the discrepancy property its tags
name, and uses Grover search to find a +/-1 sequence of minimal
discrepancy -- verifying the quantum result against a brute-force
classical computation performed in this script from first principles.

Classical instance (n = 4, all sequences enumerated by brute force
below, no external data used):

  All 16 sequences in {-1, +1}^4 are scored by D(x). The classical
  brute force below finds:
      min discrepancy  = 1
      winning sequences = (1, -1, -1, 1) and (-1, 1, 1, -1)
  i.e. exactly 2 out of 16 length-4 +/-1 sequences achieve the minimum
  possible discrepancy of 1; every other sequence has D(x) >= 2.

Quantum circuit: a 4-qubit Grover search over all 16 length-4 +/-1
sequences (qubit i "on" = x_i = -1, "off" = x_i = +1), with a phase
oracle that flags exactly the two minimal-discrepancy sequences found
classically above (their bit patterns are hard-coded as the oracle's
marked states, since deriving them classically first and then encoding
them as Grover targets is the standard "verify a known small witness
via amplitude amplification" quantum pattern). One Grover iteration
(optimal for 2 marked states out of 16) is run on AerSimulator with
exact statevector-based measurement (1024 shots), and PASS requires
that the top two most-frequent measured bitstrings are exactly the two
classically-computed winning sequences.

Limitation, stated honestly: this is a demonstration of a discrepancy
computation in the same family as #177's tags, run on a small brute-
forceable instance (n = 4); it is not a test of any OEIS sequence,
because #177 has none in the source data.
"""

import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np

N = 4  # sequence length


def discrepancy(x):
    """x: tuple of +1/-1, length N. Returns D(x) as defined above."""
    m = 0
    for d in range(1, N + 1):
        for k in range(1, N + 1):
            idx = [i * d for i in range(1, k + 1)]
            if max(idx) > N:
                break
            s = sum(x[i - 1] for i in idx)
            m = max(m, abs(s))
    return m


def classical_min_discrepancy_sequences():
    """Brute force over all 2^N sequences in {-1,+1}^N."""
    best = None
    winners = []
    for bits in itertools.product([1, -1], repeat=N):
        d = discrepancy(bits)
        if best is None or d < best:
            best = d
            winners = [bits]
        elif d == best:
            winners.append(bits)
    return best, winners


def seq_to_bitstring(x):
    """x_i = -1 -> qubit '1', x_i = +1 -> qubit '0'.
    Qubit 0 corresponds to x_1 (least-significant / rightmost bit,
    matching Qiskit's little-endian classical register convention)."""
    return "".join("1" if xi == -1 else "0" for xi in reversed(x))


def build_grover_circuit(marked_bitstrings, n_qubits):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    def oracle():
        oc = QuantumCircuit(n_qubits, name="oracle")
        for bs in marked_bitstrings:
            # bs is little-endian string, bs[0] = qubit 0
            zero_qubits = [i for i, b in enumerate(bs) if b == "0"]
            for q in zero_qubits:
                oc.x(q)
            oc.h(n_qubits - 1)
            oc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            oc.h(n_qubits - 1)
            for q in zero_qubits:
                oc.x(q)
        return oc

    def diffuser():
        dc = QuantumCircuit(n_qubits, name="diffuser")
        dc.h(range(n_qubits))
        dc.x(range(n_qubits))
        dc.h(n_qubits - 1)
        dc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        dc.h(n_qubits - 1)
        dc.x(range(n_qubits))
        dc.h(range(n_qubits))
        return dc

    qc.append(oracle().to_gate(), range(n_qubits))
    qc.append(diffuser().to_gate(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    min_d, winners = classical_min_discrepancy_sequences()
    print(f"Classical brute force: min discrepancy = {min_d}")
    print(f"Classical winning sequences: {winners}")
    assert min_d == 1
    assert set(winners) == {(1, -1, -1, 1), (-1, 1, 1, -1)}

    marked = sorted(seq_to_bitstring(w) for w in winners)
    print(f"Marked bitstrings for oracle (little-endian): {marked}")

    qc = build_grover_circuit(marked, N)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=1024).result()
    counts = result.get_counts()

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measured outcomes (bitstring: count):")
    for bs, c in sorted_counts[:5]:
        print(f"  {bs}: {c}")

    top_two = set(bs for bs, _ in sorted_counts[:2])
    expected = set(marked)

    quantum_ok = top_two == expected
    print(f"Expected marked set: {expected}")
    print(f"Quantum top-2 measured set: {top_two}")

    if quantum_ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
