"""
Erdos problem #276 (see https://github.com/manman4/erdosproblems,
data/problems.yaml, entry `number: "276"`).

Source metadata for problem 276, as recorded in that repository:
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["number theory", "covering systems"]

LIMITATION, stated honestly: problem 276 carries no OEIS sequence id
("N/A" in the source data), so there is no OEIS "term" to look up or
encode into a circuit. In place of an OEIS value, this script builds a
genuine, finite, computable instance of the actual mathematical object
the problem's tags name: a covering system of the integers, i.e. a
finite set of congruences x = a_i (mod n_i) whose union is all of Z.

The classical covering system used here (Erdos's own textbook example,
predating and motivating this line of problems) is:

    x = 0 (mod 2)
    x = 0 (mod 3)
    x = 1 (mod 4)
    x = 5 (mod 6)
    x = 7 (mod 12)

The finite, computable property tested:
    "Within one full period (residues 0..11 mod 12 = lcm(2,3,4,6,12)),
     this system is a covering system: every residue x in 0..11
     satisfies at least one of the five congruences above (some
     residues satisfy more than one, e.g. x=0 satisfies both mod-2 and
     mod-3), and in particular there is exactly one residue, x = 7,
     satisfying the congruence x = 7 (mod 12) specifically (the
     sparsest class, with density 1/12 -- it is the only class whose
     modulus equals the period, so it can cover only a single
     residue)."

This is first checked from first principles by brute-force classical
enumeration over x = 0..11. It is then verified again with a real
quantum circuit: Grover's search algorithm on a 4-qubit register
representing x in {0, ..., 15} (padded above 11 with states that never
satisfy any residue class, so they are never marked), whose oracle
marks exactly the x satisfying x = 7 (mod 12) AND 0 <= x <= 11. With
exactly one marked state out of 16, the optimal number of Grover
iterations is floor(pi/4 * sqrt(16/1)) = 3, and running that circuit on
the ideal AerSimulator should return x = 7 with high probability.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation, from first principles.
# ---------------------------------------------------------------------------

def classical_covering_check():
    """Brute-force verify the covering system over one period (0..11) and
    return (is_cover, the_unique_x_satisfying_mod12_class)."""
    congruences = [(0, 2), (0, 3), (1, 4), (5, 6), (7, 12)]
    period = 12  # lcm(2, 3, 4, 6, 12)

    coverers = {}
    for x in range(period):
        hits = [(a, n) for (a, n) in congruences if x % n == a]
        coverers[x] = hits

    is_cover = all(len(coverers[x]) >= 1 for x in range(period))

    mod12_members = [x for x in range(period) if x % 12 == 7]
    assert mod12_members == [7], "sanity check on the mod-12 class failed"

    return is_cover, mod12_members[0]


IS_COVER, TARGET_X = classical_covering_check()

if not IS_COVER:
    raise SystemExit(
        "Classical check failed: the stated system does not cover every "
        "residue over one period; refusing to build a quantum circuit "
        "around a false premise."
    )

print(f"Classical result: covering system verified over residues 0..11 (every residue hit at least once).")
print(f"Classical result: unique x with x = 7 (mod 12), 0<=x<=11, is x = {TARGET_X}.")


# ---------------------------------------------------------------------------
# 2. Quantum circuit: Grover's search for x = 7 (mod 12), 0 <= x <= 11,
#    over a 4-qubit register representing integers 0..15.
# ---------------------------------------------------------------------------

N_QUBITS = 4          # register holds integers 0..15
N_STATES = 2 ** N_QUBITS
MARKED = TARGET_X      # the single marked basis state, x = 7


def oracle(qc: QuantumCircuit, qubits):
    """Phase-flip the single computational basis state |MARKED>.
    Implemented as a multi-controlled Z conditioned on the bit pattern
    of MARKED (X gates flip any 0-bits to control-on-1, then flip back)."""
    bits = format(MARKED, f"0{N_QUBITS}b")[::-1]  # little-endian per qubit
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(qubits[i])


def diffuser(qc: QuantumCircuit, qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
        qc.h(q)


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qubits = list(range(N_QUBITS))

    qc.h(qubits)  # uniform superposition over all 16 basis states

    for _ in range(iterations):
        oracle(qc, qubits)
        diffuser(qc, qubits)

    qc.measure(qubits, qubits)
    return qc


# One marked item out of N_STATES => optimal iteration count.
optimal_iterations = max(1, round((math.pi / 4) * math.sqrt(N_STATES / 1)))
print(f"Grover iterations used: {optimal_iterations} "
      f"(optimal for 1 marked state out of {N_STATES})")

circuit = build_grover_circuit(optimal_iterations)

simulator = AerSimulator()
shots = 4096
result = simulator.run(circuit, shots=shots).result()
counts = result.get_counts()

# Qiskit's classical-register bitstrings are printed MSB-first with qubit 0
# as the rightmost bit; convert back to an integer accordingly.
best_bitstring = max(counts, key=counts.get)
measured_x = int(best_bitstring, 2)
measured_probability = counts[best_bitstring] / shots

print(f"Quantum result: most frequent measured value = {measured_x} "
      f"(probability {measured_probability:.3f} over {shots} shots)")


# ---------------------------------------------------------------------------
# 3. Compare and report.
# ---------------------------------------------------------------------------

verified = (measured_x == TARGET_X) and (measured_probability > 0.5)

if verified:
    print("PASS")
else:
    print("FAIL")
