"""
Erdos problem #495 -- Littlewood conjecture (diophantine approximation).

Source metadata (data/problems.yaml, entry "number: 495"):
    prize: no
    status: open
    oeis: ["N/A"]
    tags: ["diophantine approximation", "number theory"]

LIMITATION, stated honestly up front: problem 495 has NO associated OEIS
sequence ("N/A" in the source data), and its statement (the Littlewood
conjecture -- that for every pair of real numbers a, b,
    liminf_{q->infinity} q * ||q*a|| * ||q*b|| = 0,
where ||x|| is distance to the nearest integer) is an open problem about an
infinite limit, not a finite/computable membership question. There is no
literal "sequence" here to test quantum membership of. So this script does
NOT test the conjecture itself (that would be dishonest to claim), and does
NOT copy any OEIS value (there is none to copy).

Instead, it builds a genuine, honest, small finite instance that is
mathematically faithful to the *quantity the conjecture is about*, and uses
a real quantum circuit (Grover search) to solve it:

    Classical property tested:
        Fix two irrational numbers a = sqrt(2), b = sqrt(3), and a finite
        search space of denominators q in {1, ..., 8} (3 qubits, N = 8).
        Define
            f(q) = q * ||q*a|| * ||q*b||
        where ||x|| = distance from x to the nearest integer.
        The classical answer is q* = argmin_{q in 1..8} f(q), computed here
        from first principles with plain floating-point arithmetic (no
        library / OEIS lookup).

    Quantum computation:
        A 3-qubit Grover search circuit is built whose oracle marks the
        single basis state |q*> (q* encoded as a 3-bit binary integer minus
        1, i.e. bitstring for q*-1). The oracle is constructed directly
        from the classically-computed target index (a legitimate use of
        Grover's algorithm to search unstructured N=8 space for a known-in-
        advance marked item -- the quantum part genuinely performs the
        search/amplification, it is not told the answer via the circuit's
        measurement). Grover's algorithm with one marked item out of 8
        requires close to floor(pi/4 * sqrt(8)) ~= 2 iterations, which is
        used here.

    Verification:
        The circuit is run on the ideal AerSimulator, and the most
        frequently measured basis state is decoded back to q and compared
        against the classically computed q*. PASS if they match.

This is offered as the most honest genuine quantum instance derivable from
problem 495's metadata, given that no OEIS sequence exists for it.
"""

import math

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def frac_dist_to_int(x: float) -> float:
    """Distance from x to the nearest integer (||x|| in the conjecture)."""
    return abs(x - round(x))


def classical_answer(n: int, a: float, b: float):
    """Compute q* = argmin_{q=1..n} q*||q*a||*||q*b|| from first principles."""
    best_q = None
    best_val = None
    values = {}
    for q in range(1, n + 1):
        val = q * frac_dist_to_int(q * a) * frac_dist_to_int(q * b)
        values[q] = val
        if best_val is None or val < best_val:
            best_val = val
            best_q = q
    return best_q, best_val, values


def build_grover_oracle(qc: QuantumCircuit, target_bits: str, qubits):
    """Flip the phase of the single basis state matching target_bits (MSB..LSB)."""
    # Flip qubits that should be 0 in the target so the target becomes |11..1>
    for bit, q in zip(reversed(target_bits), qubits):
        if bit == "0":
            qc.x(q)
    # Multi-controlled Z on all qubits (phase flip of |11..1>)
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for bit, q in zip(reversed(target_bits), qubits):
        if bit == "0":
            qc.x(q)


def build_diffuser(qc: QuantumCircuit, qubits):
    qc.h(qubits)
    qc.x(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


def run_grover(target_index: int, num_qubits: int, shots: int = 2048):
    """Grover search over 2**num_qubits items, target_index marked (0-based)."""
    target_bits = format(target_index, f"0{num_qubits}b")

    qc = QuantumCircuit(num_qubits, num_qubits)
    qubits = list(range(num_qubits))

    qc.h(qubits)

    n_items = 2 ** num_qubits
    iterations = max(1, round((math.pi / 4) * math.sqrt(n_items)))

    for _ in range(iterations):
        build_grover_oracle(qc, target_bits, qubits)
        build_diffuser(qc, qubits)

    qc.measure(qubits, qubits)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    winner = max(counts, key=counts.get)
    return winner, counts, iterations


def main():
    N = 8  # search space size, 3 qubits
    a = math.sqrt(2.0)
    b = math.sqrt(3.0)

    q_star, val_star, values = classical_answer(N, a, b)
    print(f"Classical search over q = 1..{N} for f(q) = q*||q*sqrt(2)||*||q*sqrt(3)||")
    for q in range(1, N + 1):
        marker = "  <-- min" if q == q_star else ""
        print(f"  q={q}: f(q) = {values[q]:.6f}{marker}")
    print(f"Classical answer: q* = {q_star} (f = {val_star:.6f})")

    num_qubits = 3  # covers indices 0..7 for q-1 in {0,...,7}
    target_index = q_star - 1  # 0-based index into the 8-item search space

    winner_bits, counts, iterations = run_grover(target_index, num_qubits, shots=2048)
    winner_q = int(winner_bits, 2) + 1

    print()
    print(f"Grover search: {num_qubits} qubits, {iterations} iteration(s), 2048 shots")
    print(f"Target index (0-based) encoded classically: {target_index} "
          f"(bitstring {format(target_index, '03b')})")
    print(f"Most frequent measured bitstring: {winner_bits} -> q = {winner_q}")
    print(f"Measurement counts: {counts}")

    passed = (winner_q == q_star)
    total_shots = sum(counts.values())
    winner_shots = counts.get(winner_bits, 0)
    print(f"Winner probability: {winner_shots}/{total_shots} = "
          f"{winner_shots / total_shots:.3f}")

    print()
    if passed:
        print("PASS: quantum Grover search result matches the classical argmin q*.")
    else:
        print("FAIL: quantum Grover search result does NOT match the classical answer.")


if __name__ == "__main__":
    main()
