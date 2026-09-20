"""
Erdos problem #445 -- quantum-testable sequence lane.

LIMITATION (read first): the source metadata for problem #445 in
manman4/erdosproblems (data/problems.yaml, entry "number: \"445\"") lists
    oeis: ["N/A"]
    tags: ["number theory"]
    informal_status: open, last_update 2025-08-31
i.e. there is no OEIS sequence id attached to this problem at all, and no
problem statement text is present in the read-only clone to derive one from
first principles. There is therefore no genuine OEIS-linked sequence for
this lane to build a circuit around, and fabricating one would violate the
task's own instruction not to invent mathematical content that isn't there.

Per the task's fallback instruction ("if no genuine quantum circuit can be
constructed ... write the script anyway with your best honest attempt, note
the limitation clearly, and report accurately"), this script instead builds
a REAL, small, self-contained quantum circuit on a genuine, independently
checkable finite number-theory property -- primality -- run as a Grover
search, since problem #445's own tag is "number theory" and this is the
smallest honest stand-in that still exercises real quantum search machinery
rather than a fabricated "sequence".

Concrete classical property tested (computed from first principles below,
not copied from any table): among the 4-bit integers N = 0..15, find the
subset that is prime according to trial division. That classical answer is
computed in this script, and a Grover search circuit (built purely from
X/MCX gates implementing an oracle that flags exactly the primes in 0..15,
plus the standard diffusion operator) is run on the ideal AerSimulator and
its most-frequent measured outcomes are compared against the classical
prime set for PASS/FAIL.

This is NOT a claim that A-does-not-exist "is" an OEIS sequence for problem
445 -- it explicitly is not, and ran_ok / verified_against_classical are
reported honestly against this substitute construction, not against any
OEIS id for #445.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_primes_4bit():
    """Trial-division primality over N = 0..15, computed from scratch."""
    primes = []
    for n in range(16):
        if n < 2:
            continue
        is_p = True
        for d in range(2, int(math.isqrt(n)) + 1):
            if n % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(n)
    return primes


def oracle_circuit(marked_values, n_qubits):
    """Phase oracle: flips sign of |x> for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for val in marked_values:
        bits = format(val, f"0{n_qubits}b")[::-1]  # little-endian
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


def diffusion_circuit(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffusion")
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


def build_grover(marked_values, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle = oracle_circuit(marked_values, n_qubits)
    diffusion = diffusion_circuit(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffusion, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    n_qubits = 4
    N = 2 ** n_qubits  # 16

    classical = classical_primes_4bit()
    print(f"Classical primes in 0..{N - 1} (trial division): {classical}")

    M = len(classical)
    # Optimal Grover iteration count for M marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M) - 0.5))
    print(f"Running Grover search with {iterations} iteration(s) for M={M}, N={N}")

    qc = build_grover(classical, n_qubits, iterations)

    sim = AerSimulator()
    shots = 4096
    job = sim.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts()

    # Qiskit's count keys are standard binary strings "c[n-1]...c1c0" whose
    # integer value already matches the computational-basis index used above
    # (qubit 0 is the least-significant bit, c0 is the rightmost character).
    value_counts = Counter()
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        value_counts[val] += c

    # Take the top-M measured values as the quantum-found "prime set".
    top_values = sorted(v for v, _ in value_counts.most_common(M))
    quantum_hits = sum(value_counts[v] for v in classical)
    hit_fraction = quantum_hits / shots

    print(f"Top-{M} most frequently measured values: {top_values}")
    print(f"Fraction of shots landing on a true prime: {hit_fraction:.3f}")

    verified = (set(top_values) == set(classical)) and hit_fraction > 0.5
    print(f"Classical primes set: {sorted(classical)}")
    print(f"Quantum-found set:    {top_values}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
