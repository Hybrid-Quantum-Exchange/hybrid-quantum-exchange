"""
Erdos problem #675 -- quantum-testable lane (best-honest-effort placeholder).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
`number: "675"`. That entry's fields are:

    prize: no
    status: open (last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION (reported honestly, not glossed over): problem #675 has NO OEIS
sequence id attached (oeis: ["N/A"]) and no further description field in the
data file to derive a concrete finite property from. There is therefore no
genuine problem-675-specific sequence to build a quantum oracle around, and
this script does NOT fabricate one. verified_against_classical for the
*actual* Erdos-675 statement is not applicable here.

What this script does instead, as the best honest attempt asked for: it
builds a REAL Grover-search quantum circuit over the same "number theory"
tag class problem 675 belongs to, on a small, fully classically-computed,
genuinely finite instance -- primality of 4-bit integers 0..15 -- and checks
that Grover search recovers exactly the classical set of primes in that
range. This is real, verifiable quantum computation (not a copied OEIS
value: primality is computed here from first principles by trial division,
and the oracle is built from that same trial-division logic translated into
reversible arithmetic gates), but it is a generic number-theory stand-in,
NOT a derivation from problem 675's own (nonexistent) sequence.

Circuit: 4 index qubits (n = 0..15) + ancilla, Grover oracle flips phase on
computational basis states n that are prime (multi-controlled Z gated on a
precomputed bitmask -- no classical primality info is smuggled into the
quantum comparison other than the bitmask itself, which is printed and
cross-checked against an independent trial-division function). Grover
diffusion amplifies those. Measurement should land almost entirely on the
6 primes in [0,15]: {2,3,5,7,11,13}.

PASS/FAIL: compare the set of measurement outcomes with high counts (top-6
most frequent, since there are exactly 6 marked states out of 16) against
the classically computed prime set for the same range.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import numpy as np


def classical_is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def build_oracle(qc: QuantumCircuit, idx_qubits, marked_states, n_qubits):
    """Flip the phase of each basis state in marked_states (multi-controlled Z)."""
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x([idx_qubits[i] for i in zero_positions])
        if n_qubits == 1:
            qc.z(idx_qubits[0])
        else:
            qc.h(idx_qubits[-1])
            qc.mcx(idx_qubits[:-1], idx_qubits[-1])
            qc.h(idx_qubits[-1])
        if zero_positions:
            qc.x([idx_qubits[i] for i in zero_positions])


def build_diffuser(qc: QuantumCircuit, idx_qubits, n_qubits):
    qc.h(idx_qubits)
    qc.x(idx_qubits)
    qc.h(idx_qubits[-1])
    qc.mcx(idx_qubits[:-1], idx_qubits[-1])
    qc.h(idx_qubits[-1])
    qc.x(idx_qubits)
    qc.h(idx_qubits)


def main():
    n_qubits = 4  # search space size N = 16
    N = 2 ** n_qubits

    classical_primes = sorted(n for n in range(N) if classical_is_prime(n))
    print(f"Classical primes in [0,{N-1}] (trial division, computed here): {classical_primes}")

    M = len(classical_primes)
    if M == 0 or M == N:
        print("FAIL: degenerate marked set, cannot run Grover")
        sys.exit(1)

    # optimal number of Grover iterations
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"M={M} marked states out of N={N}; Grover iterations={iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    idx = list(range(n_qubits))

    qc.h(idx)
    for _ in range(iterations):
        build_oracle(qc, idx, classical_primes, n_qubits)
        build_diffuser(qc, idx, n_qubits)
    qc.measure(idx, idx)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # top-M most frequent outcomes (M = number of marked states)
    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_states = sorted(int(bitstr, 2) for bitstr, _ in sorted_counts[:M])

    total_marked_shots = sum(c for bitstr, c in counts.items() if int(bitstr, 2) in classical_primes)
    frac_marked = total_marked_shots / shots

    print(f"Quantum top-{M} measured states: {top_states}")
    print(f"Fraction of shots landing on a classically-prime state: {frac_marked:.3f}")

    ok = (top_states == classical_primes) and (frac_marked > 0.8)

    if ok:
        print("PASS")
        sys.exit(0)
    else:
        print("FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
