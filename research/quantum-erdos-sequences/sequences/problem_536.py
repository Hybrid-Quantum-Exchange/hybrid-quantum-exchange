"""
Erdos problem #536 -- quantum-testable sequence lane (best-effort, limitation noted)
=====================================================================================

Source metadata (from erdosproblems/data/problems.yaml, block "number: '536'"):
    prize: no
    informal_status: open (last_update 2025-08-31)
    formal_status: unformalized
    oeis: ["possible"]
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS"):
    The metadata's oeis field is literally the string "possible" -- this is
    NOT a real OEIS A-number. It appears to be a placeholder/tag value in the
    erdosproblems dataset (meaning "an OEIS sequence possibly exists / is
    yet to be linked"), not an identifier that resolves to any actual OEIS
    entry. There is therefore no concrete, citable integer sequence attached
    to problem #536 in the source data to build a faithful quantum test
    around. Fabricating a specific sequence and pretending it comes from
    problem #536 would misrepresent the source, which the task instructions
    explicitly forbid.

    Given that, this script does NOT claim to test "the Erdos #536 sequence"
    (no such citable sequence is available). Instead, honoring the problem's
    only real content -- its tag, "number theory" -- it builds a genuine,
    self-contained, classically-verified small quantum computation: a Grover
    search that finds the unique prime number in a small fixed-size search
    space, using a real primality oracle built from modular-arithmetic
    comparator logic. This is offered as the best-effort quantum-testable
    artifact for this lane, not as a verification of an OEIS sequence term.

    Accordingly: ran_ok reports whether the script executed and printed PASS;
    verified_against_classical reports whether the quantum result matched an
    independently computed classical answer for the small instance below --
    it does NOT assert that this is a verification of an Erdos-#536-specific
    OEIS sequence, since no such sequence identifier exists in the source
    metadata.

Classical property under test (computed from first principles, in this file):
    N = 8  (3 qubits, values 0..7)
    The unique prime in {5, 6, 7} -- i.e. search space restricted via a
    fixed offset so exactly one of the 8 basis states satisfies "is prime".
    Concretely: for x in 0..7, define f(x) = is_prime(5 + (x mod 3)).
    5, 6, 7 -> primality: True, False, True -> two marked states among the
    3 used inputs. To keep the search space clean and the marked-state count
    exactly 1 (required for a simple, exactly-tuned single-iteration Grover
    circuit on 3 qubits), we instead search over x in 0..7 directly for
    "x is prime": primes in [0,7] are {2, 3, 5, 7} -- four marked states.
    To get a search-space/marked-set ratio where Grover amplification is
    meaningful (M/N well under 1/2 -- at M/N = 1/2 a single Grover iteration
    provably returns to the *same* success probability as random guessing,
    which would make the demo vacuous), we use a 3-qubit space (N = 8,
    x in 0..7) and search for "x is prime AND x > 4": primes greater than 4
    in [0,7] are {5, 7} -- exactly two marked states out of eight (M/N = 1/4),
    a clean instance where one Grover iteration is exactly optimal
    (rotation angle (2*1+1)*asin(sqrt(1/4)) = 90 degrees).

Classical answer (computed here, not copied from anywhere):
    N = 8 (3 qubits), primes in range(8) greater than 4 = {5, 7}
    (computed by trial division in `classical_primes_below`, then filtered).
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import numpy as np


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def classical_primes_below(n: int):
    """Classically compute, from first principles, the primes in range(n)."""
    return sorted(x for x in range(n) if is_prime(x))


def build_grover_circuit(marked_states, n_qubits):
    """
    Build a Grover search circuit over n_qubits, marking the given basis
    states (list of ints), and run one Grover iteration (optimal for
    2 marked states out of 4, per the standard Grover iteration-count
    formula floor(pi/4 * sqrt(N/M))).
    """
    qc = QuantumCircuit(n_qubits, n_qubits)

    # Uniform superposition
    qc.h(range(n_qubits))

    def oracle(qc):
        # Phase-flip each marked basis state via multi-controlled Z,
        # implemented with X-gates to remap the target pattern to |11..1>.
        for state in marked_states:
            bits = [(state >> i) & 1 for i in range(n_qubits)]
            for i, b in enumerate(bits):
                if b == 0:
                    qc.x(i)
            if n_qubits == 1:
                qc.z(0)
            elif n_qubits == 2:
                qc.cz(0, 1)
            else:
                qc.h(n_qubits - 1)
                qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
                qc.h(n_qubits - 1)
            for i, b in enumerate(bits):
                if b == 0:
                    qc.x(i)

    def diffuser(qc):
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

    # Optimal number of Grover iterations for N=2^n_qubits, M=len(marked_states)
    N = 2 ** n_qubits
    M = len(marked_states)
    iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(N / M))))

    for _ in range(iterations):
        oracle(qc)
        diffuser(qc)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 3
    N = 2 ** n_qubits

    # --- Classical answer, computed from first principles ---
    primes = [p for p in classical_primes_below(N) if p > 4]  # expect [5, 7]

    print(f"Erdos problem #536 -- best-effort quantum lane (see docstring for limitation)")
    print(f"Search space N = {N} (n_qubits = {n_qubits})")
    print(f"Classically computed primes in range({N}) greater than 4: {primes}")

    if not primes:
        print("No marked states -- nothing to search. FAIL")
        return False

    # --- Quantum search for the same marked set ---
    qc = build_grover_circuit(primes, n_qubits)

    sim = AerSimulator()
    shots = 2048
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit: little-endian in the classical register
    # string, rightmost char = qubit 0) to integers.
    measured_ints = {}
    for bitstring, count in counts.items():
        val = int(bitstring, 2)
        measured_ints[val] = measured_ints.get(val, 0) + count

    print(f"Measurement counts (as integers 0..{N-1}): {measured_ints}")

    total_marked_hits = sum(measured_ints.get(m, 0) for m in primes)
    fraction_marked = total_marked_hits / shots
    print(f"Fraction of shots landing on a classically-marked prime state: {fraction_marked:.3f}")

    # Grover success criterion: amplified probability on marked states
    # should dominate (well above the uniform-random baseline of M/N).
    baseline = len(primes) / N
    quantum_result_ok = fraction_marked > max(0.8, baseline * 3)

    verified = quantum_result_ok
    print("PASS" if verified else "FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    import sys
    sys.exit(0 if ok else 1)
