"""
Erdos problem #537 — quantum-testable sequence entry.

LIMITATION (read first): problem #537's entry in erdosproblems/data/problems.yaml
lists `oeis: ["possible"]` and `tags: ["number theory"]`. "possible" is not an
OEIS sequence id (no A-number) — it is a placeholder/status token in that data
file, not a computable sequence. There is therefore no genuine OEIS sequence
attached to this problem to build a Grover/oracle circuit around, and this
script does NOT fabricate one.

Best-honest-effort fallback: since the only real signal available is the tag
"number theory", this script builds a real, self-contained quantum circuit for
a small, well-defined, and independently-verifiable number-theory search
problem in the same spirit Erdos problems in this area tend to take (finding a
nontrivial factor of a composite integer) — Grover's algorithm searching over
all 4-bit numbers 2..14 for a proper divisor of N=15, verified against the
classical answer computed from first principles (trial division) in this
script. This is presented honestly as a stand-in construction, NOT as a
verification of problem #537's actual (nonexistent) sequence membership.

Classical property tested: for N = 15 and search space x in [2, 14] (4-bit
integers), is x a nontrivial divisor of N (i.e. 15 % x == 0)? The classical
answer, computed by trial division here: divisors of 15 in [2,14] are {3, 5}.

Circuit: Grover's algorithm over 4 qubits (search space 0..15, restricted in
the oracle to divisors of 15 within [2,14]), with the standard oracle +
diffuser construction, run on the ideal AerSimulator (statevector-based exact
simulation, no shot noise concerns beyond sampling).

PASS/FAIL: the script samples the final circuit, takes the most frequent
measured bitstrings, and checks that they are a subset of the classically
computed divisor set {3, 5} (and that the divisor set is non-empty), i.e. that
Grover's algorithm actually amplified the correct classical answers.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_divisors(n: int, lo: int, hi: int):
    """Trial division from first principles: all x in [lo, hi] with n % x == 0."""
    return sorted(x for x in range(lo, hi + 1) if x != 0 and n % x == 0)


def build_oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in `marked` (as an n_qubits bitstring)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main():
    N = 15
    LO, HI = 2, 14
    N_QUBITS = 4  # covers 0..15

    marked = classical_divisors(N, LO, HI)
    print(f"Classical (trial division) divisors of {N} in [{LO},{HI}]: {marked}")

    if not marked:
        print("FAIL: no classical divisors found, nothing to search for.")
        sys.exit(1)

    num_states = 2 ** N_QUBITS
    num_marked = len(marked)
    # optimal number of Grover iterations
    iterations = max(1, round((np.pi / 4) * np.sqrt(num_states / num_marked)))

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(N_QUBITS, marked)
    diffuser = build_diffuser(N_QUBITS)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(N_QUBITS))
        qc.append(diffuser.to_gate(), range(N_QUBITS))

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's measured bitstring is "c_{n-1}...c_1 c_0" (qubit 0 = LSB,
    # rightmost character), which is already standard binary encoding.
    decoded = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        decoded[val] = decoded.get(val, 0) + c

    sorted_results = sorted(decoded.items(), key=lambda kv: -kv[1])
    top_values = [v for v, _ in sorted_results[:num_marked]]

    print(f"Grover iterations used: {iterations}")
    print(f"Top {num_marked} measured value(s) by frequency: {top_values}")
    print(f"Full counts (decoded): {sorted(decoded.items(), key=lambda kv: -kv[1])}")

    quantum_answer = sorted(top_values)
    classical_answer = sorted(marked)

    verified = set(top_values).issubset(set(marked)) and quantum_answer == classical_answer

    if verified:
        print(f"Quantum result {quantum_answer} matches classical answer {classical_answer}: PASS")
        sys.exit(0)
    else:
        print(f"Quantum result {quantum_answer} does NOT match classical answer {classical_answer}: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
