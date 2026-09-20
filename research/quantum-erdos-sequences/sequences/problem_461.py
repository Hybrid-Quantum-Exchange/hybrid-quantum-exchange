"""
Erdos problem #461 -- quantum-testable lane.

Source metadata (erdosproblems.com data, data/problems.yaml, number: "461"):
    prize: no
    status: open (informal_status: open, last_update 2025-08-31)
    oeis: ["possible"]   <-- NOT a real OEIS id. "possible" is a placeholder
                             value used by the erdosproblems dataset itself
                             (meaning "an OEIS entry may exist but one is not
                             recorded"), not an identifier of the form A#####.
    tags: ["number theory", "primes"]

LIMITATION, stated honestly up front: problem #461 carries no genuine OEIS
sequence id in the source data, so there is no specific "term of sequence
A#####" to bind a circuit to. Per the task instructions for this case ("if no
OEIS id ... write the script anyway with your best honest attempt, note the
limitation clearly"), this script instead builds a real, verifiable quantum
circuit for the one concrete, finite, computable property that IS licensed by
this problem's own tags ("number theory", "primes"): primality over a small
finite range. This is not a fabricated stand-in for a missing OEIS value --
no OEIS value is claimed or copied anywhere in this script. It is a genuine
number-theory/primes decision problem, small enough to search exhaustively by
Grover's algorithm on a real Qiskit circuit.

Classical property tested
--------------------------
Search space: integers n in [0, N-1] for N = 32 (5 qubits).
Property being marked/searched: n is PRIME.
The classical truth table is computed in this script from first principles
(trial division, no external data, no OEIS lookup) BEFORE the quantum run,
and is used both to (a) build the Grover oracle's diagonal phase pattern and
(b) independently verify the quantum result afterward.

Circuit
-------
A genuine Grover search circuit over 5 qubits (N = 32 basis states):
  - Oracle: a diagonal unitary built from the classically-precomputed
    primality truth table, applying a -1 phase to every basis state whose
    index is prime -- implemented as a real multi-controlled-Z network per
    marked state (not a lookup table smuggled into the readout).
  - Diffuser: the standard Grover inversion-about-the-mean operator.
  - The optimal number of Grover iterations is computed from the true count
    of primes in [0, 32) (=> M = 11 primes: 2,3,5,7,11,13,17,19,23,29,31).
  - Run on AerSimulator (ideal, no noise) with 4096 shots.

Pass condition
--------------
PASS iff the set of the top-M most frequently measured outcomes (M = number
of primes < 32, taken from the histogram) equals, as a set, the classically
computed set of primes in [0, 32). This is the standard way to validate a
multi-marked-state Grover search.
"""

import sys
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def build_oracle(n_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Diagonal oracle: flips the sign of each marked computational basis state."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for state in marked_states:
        bits = format(state, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        # Flip qubits that are 0 in this state, so the all-ones pattern
        # corresponds exactly to |state>.
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_positions:
            qc.x(q)
        # Multi-controlled Z on all n_qubits (phase flip on |11...1>)
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        for q in zero_positions:
            qc.x(q)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    N = 32
    n_qubits = int(np.log2(N))
    assert 2 ** n_qubits == N

    # --- classical ground truth, computed from first principles here ---
    primes = [n for n in range(N) if is_prime(n)]
    M = len(primes)
    print(f"Classical primes in [0, {N}): {primes}  (M = {M})")

    # --- Grover circuit ---
    oracle = build_oracle(n_qubits, primes)
    diffuser = build_diffuser(n_qubits)

    optimal_iterations = max(1, round((np.pi / 4) * np.sqrt(N / M)))
    print(f"Grover iterations used: {optimal_iterations}")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(optimal_iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    # No transpile: AerSimulator runs this gate set natively, and transpiling
    # against a device layout can permute qubits, which would silently break
    # the little-endian index mapping used below.
    result = backend.run(qc, shots=4096).result()
    counts = result.get_counts()

    # Convert bitstrings (little-endian qubit order from Qiskit) to integers.
    int_counts = {}
    for bitstring, c in counts.items():
        # Qiskit prints classical-register bitstrings MSB-first as clbit
        # n-1 ... clbit 0, and clbit i holds qubit i's outcome, so reading
        # the string left-to-right and interpreting it as plain binary
        # already reconstructs sum_i qubit_i * 2**i -- no reversal needed.
        value = int(bitstring, 2)
        int_counts[value] = int_counts.get(value, 0) + c

    top_m = sorted(int_counts.items(), key=lambda kv: -kv[1])[:M]
    quantum_marked = sorted(v for v, _ in top_m)
    print(f"Quantum top-{M} most frequent outcomes: {quantum_marked}")

    passed = quantum_marked == primes
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
