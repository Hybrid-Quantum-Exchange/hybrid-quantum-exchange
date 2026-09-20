"""
Erdos problem #485 -- quantum-testable sequence attempt (LIMITATION NOTICE).

Source record: /home/user/manman4/erdosproblems/data/problems.yaml, entry
"number: \"485\"":
    prize: no
    informal_status: proved (2025-08-31)
    oeis: ["possible"]
    tags: ["analysis", "polynomials"]

LIMITATION: problem #485 has no real OEIS sequence id. The "oeis" field in
the source data is the literal string "possible", not an OEIS A-number
(e.g. "A000045") -- it is not a lookup key into OEIS at all, and the
problem's tags ("analysis", "polynomials") describe a statement about
polynomial/analytic behaviour rather than an integer sequence with a small,
finite membership/search property that a toy circuit could faithfully
encode. Web access to read the full problem statement at
erdosproblems.com is not available in this environment, and fabricating an
OEIS id or a "classical answer" not actually tied to problem 485 would
violate the task's own instruction not to fabricate content.

Per the task's fallback instructions, this script is still a genuine,
running, self-contained Qiskit program -- but the finite property it
verifies is a generic, honestly-labeled placeholder (NOT derived from
problem 485's actual mathematical content), included only to demonstrate
a real quantum computation with a classically-checked answer, since no
genuine sequence-derived property could be responsibly constructed for
this problem from the available metadata.

Placeholder property actually computed and verified below:
    Grover search over 4-bit integers n in [0, 15] for the unique n such
    that n is prime AND n mod 4 == 3 (i.e. the unique such n in this
    range: 3, 7, 11 are prime and == 3 mod 4, but the search is narrowed
    to the unique n in [0,15] additionally satisfying n > 10, giving
    n = 11). This is computed classically first (brute force, first
    principles), then found via a real Grover's algorithm circuit
    (oracle + diffuser) run on the ideal AerSimulator, and the top
    measured bitstring is compared against the classical answer.

Report accurately: ran_ok reflects whether this script executes and
prints PASS; it does NOT certify that the property is meaningful for
Erdos problem #485, which it is not.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 4  # search space size 2^4 = 16


def is_prime(k: int) -> bool:
    if k < 2:
        return False
    for d in range(2, int(k ** 0.5) + 1):
        if k % d == 0:
            return False
    return True


def classical_answer():
    """Brute-force, from first principles: unique n in [0,15] with n prime,
    n mod 4 == 3, and n > 10."""
    hits = [
        n for n in range(2 ** N_QUBITS)
        if is_prime(n) and n % 4 == 3 and n > 10
    ]
    assert len(hits) == 1, f"expected a unique target, got {hits}"
    return hits[0]


def build_oracle(target: int, n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle marking the computational basis state |target>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    bits = format(target, f"0{n_qubits}b")[::-1]  # little-endian qubit order

    # Flip qubits that should be 0 in the target, so target maps to |1111...>
    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)

    # Multi-controlled Z on all qubits (phase flip when all controls are |1>)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    for i, b in enumerate(bits):
        if b == "0":
            qc.x(i)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(target: int, n_qubits: int, shots: int = 2048):
    n_states = 2 ** n_qubits
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_states)))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(target, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    compiled = transpile(qc, sim)
    result = sim.run(compiled, shots=shots).result()
    counts = result.get_counts()

    top_bitstring = max(counts, key=counts.get)
    # Qiskit classical-bit string already has qubit 0 as the rightmost
    # (least-significant) character, matching normal binary reading.
    measured = int(top_bitstring, 2)
    return measured, counts


def main():
    target = classical_answer()
    measured, counts = run_grover(target, N_QUBITS)

    top_count = counts[max(counts, key=counts.get)]
    total = sum(counts.values())
    confidence = top_count / total

    print(f"Erdos problem #485 -- quantum lane (LIMITATION: placeholder property; see docstring)")
    print(f"Classical answer (brute force): n = {target}  "
          f"(prime, n mod 4 == 3, n > 10)")
    print(f"Grover measured (most frequent outcome): n = {measured}  "
          f"(confidence {confidence:.2%} over {total} shots)")

    if measured == target and confidence > 0.5:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
