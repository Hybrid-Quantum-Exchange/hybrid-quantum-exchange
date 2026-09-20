"""
Erdos problem #589 -- quantum-testable sequence attempt (LIMITATION NOTICE).

Source record (from erdosproblems.com's data, cross-checked in
/home/user/manman4/erdosproblems/data/problems.yaml, block starting
'- number: "589"'):

    number: "589"
    prize: "no"
    informal_status.state: "open"
    oeis: ["possible"]
    tags: ["geometry"]

"possible" is a placeholder the erdosproblems.com dataset uses for problems
that do NOT have a concrete OEIS sequence id assigned -- it is not an OEIS
id (there is no OEIS A-number here). Problem 589 is an open geometry
problem (no known formula, no finite decidable characterization given in
the record) with no attached integer sequence at all. That means the task
this script was asked to do -- take the problem's OEIS sequence id(s) and
build a small finite/computable property of *that sequence* for a quantum
circuit to test -- has no real object to work from for #589: there is no
OEIS id, and geometry problems of this kind (open, no sequence) are not by
themselves a small finite computable property.

Per instructions, rather than fabricate a fake "OEIS value" or invent a
property with no connection to problem 589, this script is submitted as an
honest best-effort placeholder:

  - ran_ok: True (the script below runs and prints PASS)
  - verified_against_classical: True, but ONLY for the toy circuit chosen
    below, which is NOT mathematically derived from Erdos problem #589 or
    from any OEIS sequence tied to it. There is no such sequence to derive
    it from.

To still deliver a *genuine* quantum computation (as instructed, "write
the script anyway with your best honest attempt"), the script below runs a
real, correctly-verified Grover search circuit on the ideal AerSimulator
for a small generic decidable arithmetic property (perfect squares among
integers 0..15, verified classically first). This demonstrates the same
class of technique (oracle + Grover diffusion, exact-vs-quantum comparison)
that would be used for a genuine Erdos-problem sequence property, but the
property itself is NOT claimed to represent problem 589's actual
mathematical content, precisely because problem 589 supplies no sequence
to search over.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_perfect_squares(n_bits: int) -> list[int]:
    """Classically find all perfect squares in [0, 2**n_bits - 1]."""
    N = 2 ** n_bits
    hits = []
    for x in range(N):
        r = int(round(x ** 0.5))
        if r * r == x:
            hits.append(x)
    return hits


def build_oracle(n_bits: int, targets: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each target computational basis state."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for t in targets:
        bits = format(t, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def main() -> bool:
    n_bits = 4  # search space size N = 16 (fits the N <= ~64 instruction)
    N = 2 ** n_bits

    # 1. Classical ground truth, derived from first principles (no OEIS
    #    lookup, no external data): perfect squares in [0, 15].
    targets = classical_perfect_squares(n_bits)
    print(f"Classical perfect squares in [0, {N - 1}]: {targets}")
    assert targets == [0, 1, 4, 9], "classical computation sanity check failed"

    M = len(targets)
    # optimal number of Grover iterations for M marked items out of N,
    # using the exact formula (rounding (pi/4)*sqrt(N/M) directly can
    # overshoot by one iteration for small N/M).
    theta = np.arcsin(np.sqrt(M / N))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))
    print(f"N={N}, M={M} marked states, Grover iterations={iterations}")

    oracle = build_oracle(n_bits, targets)
    diffuser = build_diffuser(n_bits)

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical bitstrings are already MSB-first (leftmost char =
    # highest qubit index), which matches our qubit-i-holds-bit-i-of-t
    # convention used in build_oracle, so no reversal is needed here.
    quantum_hits = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        quantum_hits[val] = quantum_hits.get(val, 0) + c

    # Take the top-M most frequent outcomes as the quantum search result.
    ranked = sorted(quantum_hits.items(), key=lambda kv: -kv[1])
    quantum_targets = sorted(v for v, _ in ranked[:M])

    total_marked_prob = sum(quantum_hits.get(t, 0) for t in targets) / shots
    print(f"Quantum top-{M} outcomes: {quantum_targets}")
    print(f"Total measured probability on true marked states: {total_marked_prob:.3f}")

    verified = (quantum_targets == sorted(targets)) and (total_marked_prob > 0.9)
    return verified


if __name__ == "__main__":
    ok = main()
    if ok:
        print("PASS")
    else:
        print("FAIL")
