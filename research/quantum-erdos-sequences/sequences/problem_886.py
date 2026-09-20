"""
Erdos problem #886 -- quantum-testable instance
================================================

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 886"):
    prize: no
    status: open (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory", "divisors"]

LIMITATION, stated honestly up front: problem #886 carries no OEIS sequence id
in the source data (oeis: ["N/A"]). There is therefore no specific integer
sequence from OEIS to build a quantum-testable membership/term property from.
The only usable signal is the problem's *tags*: "number theory", "divisors".

Rather than fabricate an OEIS value or fake a property, this script builds a
genuine, finite, computable property that matches the problem's actual
subject matter (divisors of an integer) and tests it with a real Grover
search circuit on Qiskit's AerSimulator. Nothing here claims to resolve or
even formally represent Erdos problem #886 itself -- it is an honest
best-effort quantum-testable artifact keyed to the problem's tags, in the
absence of an OEIS id to derive a property from.

Classical property under test
------------------------------
Fix N = 15. Search space: all 4-bit integers x in [0, 15].
Property P(x): x is a nontrivial divisor of N, i.e. 2 <= x <= N-1 and x | N.

For N = 15 the classical (trial division, computed in this script) answer is:
    nontrivial divisors of 15 in [0,15] = {3, 5}

Quantum approach
-----------------
Grover's algorithm over 4 qubits (search space size 16), with a phase oracle
that flags exactly the classically-precomputed marked basis states {3, 5}
(binary 0011 and 0101), followed by the standard Grover diffusion operator.
The number of Grover iterations is chosen from the standard formula
floor(pi/4 * sqrt(2^n / M)) for n=4 qubits, M=2 marked states.

The circuit is run on AerSimulator (ideal, no noise) and the most probable
measured bitstrings are compared against the classically-computed marked set
{3, 5}. PASS iff Grover's algorithm surfaces exactly and only that set as the
dominant, amplified outcomes.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def classical_nontrivial_divisors(n: int) -> set:
    """Trial division: nontrivial divisors of n in [2, n-1]."""
    return {d for d in range(2, n) if n % d == 0}


def build_oracle(n_qubits: int, marked: list) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="Diffuser")
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


def run_grover(n_qubits: int, marked: list, iterations: int, shots: int = 4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    qc = qc.decompose().decompose()

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    N = 15
    n_qubits = 4  # search space [0, 15]

    # --- classical ground truth, computed from first principles ---
    marked_classical = classical_nontrivial_divisors(N)
    print(f"Classical nontrivial divisors of {N} in [0,{2**n_qubits - 1}]: "
          f"{sorted(marked_classical)}")

    marked_list = sorted(marked_classical)
    M = len(marked_list)
    iterations = max(1, round((np.pi / 4) * np.sqrt((2 ** n_qubits) / M)))
    print(f"Grover iterations used: {iterations}")

    counts = run_grover(n_qubits, marked_list, iterations)

    total_shots = sum(counts.values())
    # bitstrings from qiskit are big-endian in the printed key but little-endian
    # in qubit order; measure(range,range) keeps qubit i -> classical bit i,
    # and Qiskit prints classical bits MSB-first, so int(key, 2) recovers the
    # integer directly since our oracle/diffuser also treat qubit 0 as LSB.
    scored = sorted(
        ((int(bitstring, 2), c) for bitstring, c in counts.items()),
        key=lambda t: -t[1],
    )
    print("Top measured outcomes (value: count):")
    for val, c in scored[:6]:
        print(f"  {val:2d} ({c/total_shots:5.1%})")

    # Dominant outcomes = top-M measured values by count.
    top_values = {val for val, _ in scored[:M]}

    # amplification sanity check: marked states should collectively carry much
    # more probability mass than a uniform-random guess would (M/2^n).
    marked_mass = sum(c for v, c in scored if v in marked_classical) / total_shots
    baseline_mass = M / (2 ** n_qubits)

    verified = (
        top_values == marked_classical
        and marked_mass > 3 * baseline_mass
    )

    print(f"Quantum top-{M} outcomes: {sorted(top_values)}")
    print(f"Marked-state probability mass: {marked_mass:.3f} "
          f"(uniform baseline: {baseline_mass:.3f})")
    print(f"Classical answer:            {sorted(marked_classical)}")

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
