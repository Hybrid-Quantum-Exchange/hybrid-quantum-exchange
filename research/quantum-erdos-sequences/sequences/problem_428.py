"""
Erdos problem #428 -- quantum-testable lane.

Source metadata (erdosproblems.com data, data/problems.yaml, entry `number: "428"`):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "primes"]

LIMITATION, stated honestly up front: problem #428 carries no OEIS sequence id
(`oeis: ["N/A"]`) and the local clone of erdosproblems.com does not include a
prose statement of the problem alongside the YAML metadata entry. There is
therefore no specific integer sequence to target for this lane. Rather than
fabricate an OEIS id or invent an unrelated "known term" to match against, this
script instead builds a real, checkable quantum circuit for the one concrete,
finite, computable property the metadata *does* license: the problem's own
"primes" tag. The classical property under test is genuine number theory
(primality over a finite search space), and the quantum circuit is a real
Grover search over that space -- but the link to problem #428 specifically is
only through its tag, not through a sequence unique to that problem. This is
reported honestly below rather than claimed as a verified OEIS-derived result.

Classical property tested
--------------------------
Over the 4-bit search space N = {0, 1, ..., 15}, let S = { n in N : n is prime }.
Trial division (computed in this script, from first principles, no lookup
table) gives:
    S = {2, 3, 5, 7, 11, 13}   (|S| = 6 of 16)

Quantum circuit
----------------
A standard Grover search (oracle + diffuser, amplitude amplification) is built
over 4 qubits. The oracle phase-flips exactly the basis states in S (marked by
trial-division-verified primality, not looked up from any table). The number
of Grover iterations is chosen optimally for |S|=6 out of N=16
(r = round(pi/4 * sqrt(N/|S|))). The circuit is run on the ideal AerSimulator
with many shots; PASS requires that the set of basis states receiving
amplified (majority) measurement probability equals S exactly, i.e. the
circuit's search output classically reconstructs the correct primality set for
this finite instance.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_prime_set(n_qubits: int):
    N = 2 ** n_qubits
    return sorted(n for n in range(N) if is_prime(n))


def build_oracle(n_qubits: int, marked_states):
    """Phase-flip each marked basis state (multi-controlled Z via ancilla-free MCX+phase)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")
        # flip 0-bits to 1 so the target state maps to |11...1>
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(n_qubits: int, marked_states, shots: int = 20000):
    N = 2 ** n_qubits
    n_iters = max(1, round((math.pi / 4) * math.sqrt(N / len(marked_states))))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked_states)
    diffuser = build_diffuser(n_qubits)
    for _ in range(n_iters):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, n_iters


def main():
    n_qubits = 4
    N = 2 ** n_qubits

    classical_S = classical_prime_set(n_qubits)
    print(f"Search space: N = {{0,...,{N-1}}} ({n_qubits} qubits)")
    print(f"Classical prime set S (trial division): {classical_S}  (|S|={len(classical_S)})")

    counts, n_iters = run_grover(n_qubits, classical_S)
    shots = sum(counts.values())
    print(f"Grover iterations used: {n_iters}")

    # Reconstruct the amplified set: states whose measured probability clearly
    # exceeds the uniform baseline 1/N, i.e. the states Grover boosted.
    baseline = 1.0 / N
    amplified = []
    for state_str, c in counts.items():
        prob = c / shots
        n = int(state_str, 2)
        if prob > 2 * baseline:
            amplified.append(n)
    amplified_set = sorted(amplified)

    print(f"Quantum-amplified set (prob > 2x uniform baseline): {amplified_set}")

    ok = amplified_set == classical_S
    print("PASS" if ok else "FAIL")

    return {
        "ran_ok": True,
        "verified_against_classical": ok,
    }


if __name__ == "__main__":
    main()
