"""
Erdos problem #332 -- LIMITATION NOTICE, read before trusting this script.

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"332\"" (line 5456). Its fields are:
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["N/A"]
    tags: ["number theory"]

Problem #332 carries NO OEIS sequence id (oeis is literally "N/A" in the
source data) and no problem-statement text file exists anywhere in the
manman4/erdosproblems clone (checked: no file matching "*332*"). The tag
"number theory" is the only content-bearing metadata available. There is
therefore no genuine, problem-332-specific integer sequence to build a
quantum-testable property from -- this is exactly the "no OEIS id" case the
task instructions anticipate, and per those instructions this script is the
best honest attempt rather than a faked pass tied to problem 332.

What this script actually does, honestly labeled as a stand-in:
It builds a REAL Grover search circuit over a small number-theoretic
property that a "number theory" Erdos problem plausibly concerns --
primality -- applied to a small finite instance with no claimed connection
to the specific (unknown) content of problem #332:

    Property tested: "n is prime", searched over n in {0, 1, ..., 15}
    (4 qubits, N = 16).

The classical answer (which n in [0,16) are prime) is computed from first
principles by trial division in `classical_primes_under_16()`, independent
of any OEIS lookup. A Grover oracle marks exactly those n, amplitude
amplification is run for the optimal number of iterations, and the
simulated measurement distribution is compared against the classical set.

Reported accurately (see harness output): this script runs and verifies
its own (stand-in) property against its own classical computation, but it
is NOT verified against problem #332's actual mathematical content, because
problem #332 has no OEIS id and no available statement text to derive one
from.
"""

import math
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# Classical ground truth, computed from first principles (trial division).
# ---------------------------------------------------------------------------
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def classical_primes_under_16():
    return sorted(n for n in range(16) if is_prime(n))


# ---------------------------------------------------------------------------
# Grover oracle: flips the phase of computational basis states |n> whose
# 4-bit binary value is prime. Built as an explicit multi-controlled-Z per
# marked value (fine for N = 16).
# ---------------------------------------------------------------------------
def build_oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian per qubit
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
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


def build_grover_circuit(n_qubits: int, marked_values, iterations: int):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 4
    N = 2 ** n_qubits  # 16

    primes = classical_primes_under_16()
    M = len(primes)

    # Optimal Grover iteration count: floor(pi/4 * sqrt(N/M))
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    qc = build_grover_circuit(n_qubits, primes, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstrings are already MSB-first over the
    # qubit index (c_{n-1}...c_0), which is exactly the integer value used
    # to build the oracle -- no reversal needed.
    measured_hist = {}
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        measured_hist[value] = measured_hist.get(value, 0) + count

    # Take the top-M most frequently measured values as the circuit's
    # answer for "which n are prime in [0,16)".
    top_values = sorted(measured_hist.items(), key=lambda kv: kv[1], reverse=True)
    quantum_answer = sorted(v for v, _ in top_values[:M])

    prob_on_marked = sum(measured_hist.get(v, 0) for v in primes) / shots

    print(f"Classical primes in [0,16): {primes}")
    print(f"Grover iterations used: {iterations}")
    print(f"Quantum top-{M} measured values: {quantum_answer}")
    print(f"Fraction of shots landing on a marked (prime) value: {prob_on_marked:.4f}")

    verified = (quantum_answer == primes) and (prob_on_marked > 0.5)

    print("PASS" if verified else "FAIL")

    print(
        "\nNOTE: this PASS/FAIL is for the stand-in primality-search circuit "
        "only. Erdos problem #332 has oeis: ['N/A'] in the source data and "
        "no statement text was found in the local clone, so this result is "
        "NOT a verification of problem #332's actual mathematical content."
    )

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
