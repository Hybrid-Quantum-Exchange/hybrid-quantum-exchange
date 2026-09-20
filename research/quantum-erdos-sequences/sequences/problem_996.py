"""
Erdos problem #996 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, data/problems.yaml in the
manman4/erdosproblems clone), entry for number "996":

    prize: no
    informal_status: open (last_update 2025-09-07)
    formal_status: unformalized
    formalized: yes (last_update 2025-12-22)
    oeis: ["N/A"]
    tags: ["analysis"]

LIMITATION (reported honestly, not papered over): problem #996 has NO OEIS
sequence attached (oeis == ["N/A"]) and its single tag, "analysis", gives no
finite combinatorial/number-theoretic object either -- the erdosproblems.com
page for #996 states an open real-analysis question about function growth
rates, which has no natural encoding as a small finite decision/search
problem that a several-qubit circuit could meaningfully compute. There is
therefore no genuine, non-fabricated finite property of "the OEIS sequence
for problem 996" to build a quantum circuit around, because no such sequence
exists for this problem.

Per the task instructions for exactly this situation, this script proceeds
with a best-effort, clearly-labeled substitute rather than inventing a fake
OEIS value or a fake connection to problem #996: it builds a REAL Grover
search circuit -- the same general-purpose technique this lane library uses
for problems that do have a finite property -- over the small, well-defined,
classically-checkable number-theoretic property "n is prime" on n in
[0, 15] (4 qubits, N = 16). This is NOT derived from problem #996's content;
it is included only so this lane still exercises and PASSes a genuine
Qiskit quantum computation, while the docstring makes plain the search space
has no mathematical connection to Erdos problem #996 itself.

Classical property under test: for N = 16 (4-bit integers 0..15), the set of
primes is {2, 3, 5, 7, 11, 13} (computed below by trial division, not looked
up). The oracle marks exactly these basis states; Grover's algorithm is run
with the standard optimal iteration count and the measured output
distribution is checked against the classical marked set.

ran_ok / verified_against_classical for this lane should be read together
with this limitation: the circuit runs and its measurement matches the
classical computation, but the mathematical *content* being tested is a
generic stand-in, not problem #996's actual (OEIS-less, non-finite) open
question.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes(n_max_exclusive: int):
    """Trial-division primality test computed from first principles."""
    primes = []
    for n in range(n_max_exclusive):
        if n < 2:
            continue
        is_prime = True
        for d in range(2, int(n ** 0.5) + 1):
            if n % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
    return primes


def build_oracle(n_qubits: int, marked_values):
    """Phase-flip oracle marking each value in marked_values (multi-controlled Z)."""
    oracle = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            oracle.x(i)
        oracle.h(n_qubits - 1)
        oracle.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        oracle.h(n_qubits - 1)
        for i in zero_positions:
            oracle.x(i)
    return oracle


def build_diffuser(n_qubits: int):
    """Standard Grover diffusion operator (inversion about the mean)."""
    diffuser = QuantumCircuit(n_qubits, name="diffuser")
    diffuser.h(range(n_qubits))
    diffuser.x(range(n_qubits))
    diffuser.h(n_qubits - 1)
    diffuser.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    diffuser.h(n_qubits - 1)
    diffuser.x(range(n_qubits))
    diffuser.h(range(n_qubits))
    return diffuser


def run_grover(n_qubits: int, marked_values, shots: int = 4096):
    n_states = 2 ** n_qubits
    n_marked = len(marked_values)

    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)

    theta = np.arcsin(np.sqrt(n_marked / n_states))
    iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_states = 2 ** n_qubits  # N = 16

    marked = classical_primes(n_states)
    print(f"Search space: integers 0..{n_states - 1} ({n_qubits} qubits, N={n_states})")
    print(f"Classical property under test: primality (trial division)")
    print(f"Classical marked set (primes): {marked}")

    counts, iterations = run_grover(n_qubits, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")

    # Sort measured outcomes by frequency, take the top len(marked) as the
    # circuit's answer for "which basis states are marked".
    def bits_to_int(bitstring: str) -> int:
        # Qiskit's classical-register bitstrings are already MSB-first
        # (leftmost char = highest classical bit index), matching the
        # standard binary encoding used when building the oracle.
        return int(bitstring, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = sorted_counts[: len(marked)]
    measured_values = sorted(bits_to_int(b) for b, _ in top)

    total_shots = sum(counts.values())
    marked_shots = sum(c for b, c in counts.items() if bits_to_int(b) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Top {len(marked)} measured basis states (by count): {measured_values}")
    print(f"Fraction of shots landing on a classically-marked (prime) state: {marked_fraction:.3f}")

    # Verification: the set of most-frequent measured states must equal the
    # classical marked set, and the amplified probability mass on marked
    # states must clearly dominate (well above the 6/16 ~ 0.375 baseline
    # of uniform sampling, confirming genuine amplitude amplification).
    sets_match = set(measured_values) == set(marked)
    amplification_confirmed = marked_fraction > 0.8

    passed = sets_match and amplification_confirmed

    print(f"Measured set matches classical marked set: {sets_match}")
    print(f"Amplitude amplification confirmed (>0.8 mass on marked states): {amplification_confirmed}")
    print("PASS" if passed else "FAIL")

    return passed


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
