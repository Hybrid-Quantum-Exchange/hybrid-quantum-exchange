"""
Erdos problem #680 (from https://github.com/manman4/erdosproblems,
data/problems.yaml, number: "680") — LIMITATION NOTE FIRST:

Problem #680's YAML entry carries oeis: ["N/A"] — no OEIS sequence id at
all — and tags ["number theory", "primes"] with no informal statement text
included in the data file. There is therefore no specific sequence to build
a faithful quantum-testable instance of. Per the task instructions, this is
the "best honest attempt" fallback: rather than fabricate a sequence or copy
an OEIS value that doesn't exist here, this script builds a genuine, real
quantum circuit for the closest well-defined, finite, computable property
implied by the problem's own tags (number theory / primes): "which integers
in a small range are prime". This is a legitimate Grover-search instance,
not a stand-in for problem #680's actual (unformalized, open) content, and
should not be read as verifying anything about problem #680 itself beyond
its stated tags.

Classical property tested (computed from first principles in this script,
no lookup table, no OEIS copy):
    For n in {0, 1, ..., 15} (4-bit unsigned integers, N = 16),
    which n are prime?
    A trial-division primality test is implemented directly below and used
    to compute the classical ground truth set of primes in [0, 15]:
        {2, 3, 5, 7, 11, 13}

Quantum approach:
    A 4-qubit Grover search whose oracle marks exactly the prime residues
    among the 16 possible 4-qubit basis states (built from the classical
    primality test above, i.e. the oracle's marked set IS the classically
    computed answer — the circuit does not need to "discover" primality
    internally, only to amplify exactly the marked classical answer set via
    genuine Grover diffusion/oracle machinery). After the computed optimal
    number of Grover iterations, the simulator is run on AerSimulator and
    the resulting measurement distribution is checked: PASS iff essentially
    all sampled measurement outcomes land in the classically-computed prime
    set (i.e. amplitude has been genuinely amplified onto the correct
    classical answer).

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def is_prime(n: int) -> bool:
    """Trial-division primality test, computed from first principles."""
    if n < 2:
        return False
    for d in range(2, int(math.isqrt(n)) + 1):
        if n % d == 0:
            return False
    return True


def build_oracle(marked, n_qubits):
    """Phase-flip oracle marking each basis state in `marked` (list of ints)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        # multi-controlled Z on all n_qubits (phase flip if all qubits are 1)
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(n_qubits):
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
    N_QUBITS = 4
    N = 2 ** N_QUBITS  # 16

    # --- classical ground truth, computed here from first principles ---
    classical_primes = [n for n in range(N) if is_prime(n)]
    print(f"Classical primes in [0, {N - 1}]: {classical_primes}")

    M = len(classical_primes)
    assert M > 0

    # optimal number of Grover iterations for N items, M marked
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Grover iterations used: {iterations}")

    oracle = build_oracle(classical_primes, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstring is c[n-1]...c[0] (qubit i = bit i,
    # i.e. qubit 0 is the LSB), so it is already the standard binary
    # representation of the integer -- no reversal needed here.
    outcome_counts = {}
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        outcome_counts[val] = outcome_counts.get(val, 0) + c

    marked_hits = sum(outcome_counts.get(p, 0) for p in classical_primes)
    fraction_on_marked = marked_hits / shots
    print(f"Measurement outcomes (value: count): {sorted(outcome_counts.items())}")
    print(f"Fraction of shots landing on classically-computed primes: {fraction_on_marked:.4f}")

    # Success threshold: Grover amplification with these parameters should
    # concentrate the large majority of amplitude on the marked (prime) set.
    threshold = 0.75
    passed = fraction_on_marked >= threshold

    if passed:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
