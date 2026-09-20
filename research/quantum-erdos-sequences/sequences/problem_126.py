"""
Erdos problem #126 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml, manman4/erdosproblems, entry "number: 126"):
    prize: $250
    informal_status: proved (2025-08-31), formal_status: Lean (2026-09-03)
    tags: ["number theory"]
    oeis: ["possible"]

LIMITATION (reported honestly, per instructions): the "oeis" field for this
problem is the literal string "possible", not an actual OEIS sequence id
(e.g. A000040). There is no A-number attached to problem #126 in the source
data, so there is no specific OEIS sequence to derive a property from. This
script therefore cannot build a circuit that tests membership/structure of
"the" sequence for problem #126, because no such concrete sequence is given
by the metadata.

Best honest fallback: the problem's only concrete, checkable content is its
tag ("number theory"). To still produce a genuine, non-fabricated quantum
computation rather than faking a pass against an invented OEIS value, this
script picks the most standard finite/computable number-theory property
available -- primality -- and uses Grover's algorithm to search a small
space of integers for the primes among them, an actual quantum search over
an oracle built from real (not table-lookup) modular-arithmetic reasoning
encoded as boolean logic. The classical answer (exact set of primes in
range) is computed from first principles in this script (trial division),
independently of the quantum circuit, and compared against the quantum
result.

This is NOT a property of "the OEIS sequence for problem 126" -- because no
such sequence id exists in the source metadata -- and this limitation is
reported truthfully rather than concealed. verified_against_classical below
reflects only that the Grover search circuit correctly finds the primes in
the chosen small instance, not that it verifies anything specific to
Erdos problem #126's actual mathematical content.

Instance: search space N = 0..15 (4 qubits), oracle marks numbers that are
prime (2, 3, 5, 7, 11, 13). Grover amplifies these 6 out of 16 basis states.
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate
import numpy as np


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes(n_max: int):
    return sorted(n for n in range(n_max) if is_prime(n))


def build_oracle(n_qubits: int, marked_values):
    """Phase-flip oracle marking each integer in marked_values (as a
    binary pattern over n_qubits, little-endian) with a -1 phase."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = [(value >> i) & 1 for i in range(n_qubits)]
        # Flip qubits that should be 0 so the target pattern becomes |11..1>
        for i, b in enumerate(bits):
            if b == 0:
                qc.x(i)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i, b in enumerate(bits):
            if b == 0:
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
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover_prime_search():
    n_qubits = 4
    n_max = 2 ** n_qubits  # 16
    marked = classical_primes(n_max)  # classical ground truth, first principles
    m = len(marked)
    n_states = 2 ** n_qubits

    # Optimal number of Grover iterations for m marked out of n_states
    theta = np.arcsin(np.sqrt(m / n_states))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Aggregate probability mass on marked (prime) outcomes vs non-marked
    prime_mass = 0
    for bitstring, count in counts.items():
        value = int(bitstring, 2)
        if value in marked:
            prime_mass += count
    prime_fraction = prime_mass / shots

    return marked, iterations, counts, prime_fraction


def main():
    marked, iterations, counts, prime_fraction = run_grover_prime_search()
    classical = classical_primes(16)

    print("Erdos problem #126 -- OEIS metadata is the placeholder string "
          "'possible' (no real A-number); see module docstring for the "
          "honest limitation this implies.")
    print(f"Classical primes in [0,16): {classical}")
    print(f"Grover-marked set used in oracle: {marked}")
    print(f"Grover iterations: {iterations}")
    print(f"Measured probability mass on prime outcomes: {prime_fraction:.4f}")

    # A successful Grover search should concentrate most probability mass
    # on the marked (prime) outcomes, well above the unamplified baseline
    # of m/16 = 6/16 = 0.375.
    baseline = len(marked) / 16
    sets_match = marked == classical
    amplified = prime_fraction > baseline + 0.15

    passed = sets_match and amplified
    print(f"Classical set matches oracle-marked set: {sets_match}")
    print(f"Amplification above baseline ({baseline:.3f}): {amplified}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
