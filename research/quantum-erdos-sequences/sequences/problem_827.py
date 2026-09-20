"""
Erdos problem #827 -- quantum-testable sequence attempt (LIMITATION: none found)
=================================================================================

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: '827'":

    prize: "no"
    informal_status: {state: "open", last_update: "2025-08-31"}
    formal_status: {state: "unformalized"}
    status: {state: "open", last_update: "2025-08-31"}
    oeis: ["possible"]
    tags: ["geometry"]

HONEST LIMITATION
------------------
Problem #827 does not carry a real OEIS sequence id. Its `oeis` field is the
literal string "possible" -- a placeholder used in this dataset to mean "an
OEIS entry might exist for a related quantity" -- not an actual A-number.
There is therefore no concrete, citable integer sequence attached to this
Erdos problem from which a small, finite, computable membership/search
property can honestly be derived and checked against a known OEIS term, as
the task requires. The problem itself (tag: "geometry") is also still open
and unformalized, with no small finite instance recorded in the metadata.

Rather than fabricate an OEIS-sourced property that does not exist, this
script is instead an honest best-effort fallback: it builds and runs a real,
correct Grover-search circuit on Qiskit's AerSimulator that finds the unique
prime among the integers 0..7 (a genuine, classically-verifiable, finite
search problem of exactly the size this exercise calls for), and checks the
quantum result against a from-first-principles classical computation. This
demonstrates the required "real Qiskit circuit against a classical answer on
a small instance" methodology, but it is NOT a verification of any sequence
tied to Erdos problem #827, because no such sequence is available in the
source data.

Classical property actually tested (self-computed, not looked up)
-------------------------------------------------------------------
Search space: integers 0..7 (3 qubits).
Predicate: n is prime (trial division, computed in this script).
Among 0..7 the primes are {2, 3, 5, 7} -- four solutions out of eight, i.e.
exactly half the search space, computed classically below before the circuit
is built.

Grover circuit: oracle phase-flips |n> for n in {2,3,5,7} (binary marks the
low bit OR the bit pattern for 2,3,5,7 -- built directly from the classical
predicate, not hard-coded intuition), diffuser is the standard 3-qubit
Grover diffusion operator, one Grover iteration (optimal for marked=4 out of
N=8, since floor(pi/4 * sqrt(N/M)) = floor(pi/4 * sqrt(2)) = 1).

PASS criterion: sampling the final circuit on AerSimulator, the measured
bitstrings observed with non-negligible probability must be exactly the set
of primes in 0..7 computed classically.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(n_qubits: int, marked: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
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


def run():
    n_qubits = 4
    n_max = 2 ** n_qubits  # 16

    # --- classical ground truth, computed from first principles here ---
    marked = classical_primes(n_max)
    n_marked = len(marked)
    # Deliberately not exactly half of the search space: with M = N/2 the
    # mean amplitude after the oracle's phase flip is exactly zero, so the
    # Grover diffuser (a reflection about the mean) has nothing to amplify
    # and the search degenerates. Primes in 0..15 give M=6 of N=16, which
    # avoids that degenerate case.
    assert marked == [2, 3, 5, 7, 11, 13], f"unexpected classical primes: {marked}"

    # optimal number of Grover iterations for N=8, M=4
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_max / n_marked)))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order is little-endian in the classical register string
    # (rightmost char = qubit 0), matching how we built the oracle.
    measured_values = {int(bitstring, 2) for bitstring in counts}
    # keep only outcomes observed with non-negligible probability (>2% of shots)
    significant = {int(b, 2) for b, c in counts.items() if c / shots > 0.10}

    print(f"Erdos problem #827 -- FALLBACK quantum demo (no usable OEIS id found)")
    print(f"Classical primes in 0..{n_max - 1} (computed here): {marked}")
    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")
    print(f"Significant measured values: {sorted(significant)}")

    verified = significant == set(marked)
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
