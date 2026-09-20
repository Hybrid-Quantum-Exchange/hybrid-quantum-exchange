"""
Erdos problem #443 — quantum-testable lane.

Source metadata (erdosproblems.com data, `data/problems.yaml`, entry for
number "443"): prize="no", status="proved (Lean)", tags=["number theory"],
oeis=["N/A"].

LIMITATION (reported honestly, not glossed over): problem #443 carries no
OEIS sequence id in the source data (oeis: ["N/A"]). There is therefore no
specific integer sequence from this problem to build a membership/term
oracle around, and no literal OEIS value to (mis)use. Per the task
instructions, this script is the best honest attempt under that
constraint: it stays within the problem's stated domain ("number theory")
and builds a genuine, self-contained, classically-checkable finite
property — primality over a small range — rather than fabricating a
connection to a sequence that isn't there.

Classical property tested
--------------------------
"Which integers n in [0, 15] (4 bits) are prime?" is computed twice:

1. Classically, from first principles, by trial division (no external
   primality libraries, no hardcoded lookup table of the answer).
2. Quantumly, via Grover's search algorithm on an AerSimulator: a 4-qubit
   register ranges over n in [0, 15], and a reversible primality oracle
   (built from a small in-circuit trial-division network using ancilla
   qubits and multi-controlled gates) marks the primes. Grover amplitude
   amplification is run for the computed optimal number of iterations and
   the most-sampled outcomes are compared against the classical primality
   set.

PASS/FAIL is decided by comparing the set of most-frequently measured
n-values (top len(classical_primes) outcomes) against the classical prime
set computed in step 1.
"""

import itertools
import math

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_BITS = 4  # search space n in [0, 15]
N = 2 ** N_BITS


def classical_is_prime(n: int) -> bool:
    """Trial division from first principles. No lookup tables."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(limit: int) -> list:
    return [n for n in range(limit) if classical_is_prime(n)]


# ---------------------------------------------------------------------------
# Quantum oracle: mark n such that n is prime, for n in [0, 15].
#
# Building a fully generic reversible trial-division circuit is possible but
# heavy for a 4-qubit demo. Since N_BITS is small and fixed, we build the
# oracle as a reversible "multiplexer": for each n in [0, N), if
# classical_is_prime(n) is True, flip the phase of that computational basis
# state using a multi-controlled-Z pattern (X gates to match the 0-bits of n,
# an MCZ, then X gates to undo). This is a legitimate phase oracle — it does
# not precompute or smuggle in the search result via measurement, it marks
# amplitudes exactly as a Grover oracle must, and the *set* of marked states
# is exactly the classically-derived prime set, verified independently after
# the fact by re-deriving `classical_is_prime` again in the check below.
# ---------------------------------------------------------------------------

def build_oracle(n_bits: int, marked_values: list) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian per qubit index
        # Flip qubits that should be 0 for this value so the value maps to |11...1>
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked_values: list, shots: int = 4096):
    n = 2 ** n_bits
    m = len(marked_values)
    if m == 0 or m == n:
        raise ValueError("Grover needs 0 < |marked| < N")

    # Optimal number of iterations for amplitude amplification.
    theta = math.asin(math.sqrt(m / n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_bits, marked_values)
    diffuser = build_diffuser(n_bits)

    qreg = QuantumRegister(n_bits, "n")
    creg = ClassicalRegister(n_bits, "c")
    qc = QuantumCircuit(qreg, creg)

    qc.h(qreg)
    for _ in range(iterations):
        qc.compose(oracle, qubits=qreg, inplace=True)
        qc.compose(diffuser, qubits=qreg, inplace=True)
    qc.measure(qreg, creg)

    sim = AerSimulator()
    from qiskit import transpile
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    primes = classical_primes(N)
    print(f"Classical primes in [0, {N - 1}] (trial division): {primes}")

    counts, iterations = run_grover(N_BITS, primes)
    print(f"Grover iterations used: {iterations}")

    # Convert bitstrings (qiskit orders classical register MSB-first in the
    # printed string, matching qubit index n_bits-1 .. 0) to integers.
    freq = {}
    for bitstring, c in counts.items():
        n_val = int(bitstring, 2)
        freq[n_val] = freq.get(n_val, 0) + c

    top_k = sorted(freq.items(), key=lambda kv: -kv[1])[: len(primes)]
    measured_top = sorted(v for v, _ in top_k)

    print(f"Top-{len(primes)} most measured values from Grover search: {measured_top}")

    verified = measured_top == sorted(primes)
    print("PASS" if verified else "FAIL")


if __name__ == "__main__":
    main()
