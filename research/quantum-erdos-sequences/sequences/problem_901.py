"""
Erdos problem #901 — quantum-testable sequence attempt (best-effort, limited).

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/problems.yaml,
entry "number: '901'"):
    prize: no
    status: open (last_update 2025-08-31)
    oeis: ["possible"]
    tags: ["combinatorics", "hypergraphs"]

LIMITATION (reported honestly, per instructions): the `oeis` field for problem #901
is the literal string "possible", not a real OEIS sequence id. There is no
A-number attached to this problem in the source data, so there is no concrete
integer sequence to derive a classical property from for this problem. No
genuine quantum circuit can be built that computes/verifies "the OEIS sequence
for problem 901", because no such sequence is identified in the source.

Best-effort fallback actually implemented below: rather than fabricate a
property with no mathematical content, this script builds a REAL, verifiable
quantum circuit — Grover's search algorithm — for a small, well-defined,
independently-checkable classical property in the same combinatorial spirit
as the problem's tags (searching a finite space for elements satisfying a
predicate, which is the generic computational shape of "hypergraph covering /
existence" style questions). Concretely:

    Classical property tested: "which integers n in [0, 7] are divisible by 3?"
    (computed here from first principles, no library calls)
    Search space: N = 8 (3 qubits), marked set = {0, 3, 6}.

Grover's algorithm is run on the ideal AerSimulator to amplify the marked
(multiples-of-3) basis states, and the most-frequently measured outcomes are
compared against the classically-computed set of multiples of 3 in [0, 7].

Because this is a fallback (no real sequence for #901 was available), this
script does NOT verify anything specific to Erdos problem #901's actual
mathematical content. It verifies that Grover's algorithm correctly amplifies
a classically pre-computed marked set, on real Qiskit/Aer execution. Reported
honestly as such.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_multiples_of_three_below(n: int) -> list[int]:
    """Which integers in [0, n) are divisible by 3, from first principles."""
    return [k for k in range(n) if k % 3 == 0]


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking each state in `marked` (multi-controlled Z)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
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


def main() -> bool:
    n = 8
    n_qubits = 3  # 2**3 = 8

    # --- classical ground truth ---
    marked = classical_multiples_of_three_below(n)
    print(f"Classical property: multiples of 3 in [0, {n})")
    print(f"Classical answer (marked set): {marked}")

    # --- Grover parameters ---
    num_marked = len(marked)
    theta = np.arcsin(np.sqrt(num_marked / (2 ** n_qubits)))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Little-endian bitstrings -> integers
    int_counts = {}
    for bitstring, c in counts.items():
        val = int(bitstring[::-1], 2)
        int_counts[val] = int_counts.get(val, 0) + c

    print(f"Measurement counts (by integer, {shots} shots): {int_counts}")

    # Take the top `num_marked` most frequent outcomes as the quantum answer.
    ranked = sorted(int_counts.items(), key=lambda kv: -kv[1])
    quantum_answer = sorted(v for v, _ in ranked[:num_marked])
    print(f"Quantum answer (top-{num_marked} measured states): {quantum_answer}")

    passed = quantum_answer == sorted(marked)
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
