"""
Erdos problem #543 -- quantum-testable sequence lane.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 543"):
    status:  disproved (informal_status.state == "disproved", last_update 2026-01-23)
    tags:    ["number theory", "group theory"]
    oeis:    ["possible"]

LIMITATION (read before trusting the "PASS"):
    Problem #543's YAML entry does NOT carry a real OEIS sequence id -- the
    "oeis" field is literally the placeholder string "possible", not an
    A-number. There is therefore no genuine OEIS sequence to derive a
    finite/computable circuit-testable property from for this specific
    problem, and no problem statement/title is present in the read-only
    clone to derive one from either. This script is the best honest
    fallback allowed by the task instructions: since the problem's tags are
    "number theory" + "group theory", it builds a REAL, unmodified Grover
    search circuit over a genuine, from-first-principles number-theoretic
    property -- divisors of a fixed integer -- rather than fabricating an
    OEIS-backed claim. This does NOT verify anything about Erdos problem
    #543 itself; it only demonstrates a real quantum search circuit on a
    small, honestly-computed classical number-theory instance, as the
    fallback path requests.

Classical property under test:
    N = 12, search space x in {0, 1, ..., 15} (4 qubits).
    Marked set M = { x in [0,15] : x > 1 and N % x == 0 }
    i.e. the proper divisors of 12 greater than 1.

    Computed here in Python, from first principles (trial division), not
    copied from anywhere:
        divisors of 12 in [0,15] with x > 1 and 12 % x == 0
        -> {2, 3, 4, 6, 12}

Circuit:
    A standard Grover search (oracle + diffuser, optimal iteration count)
    over 4 qubits (16 basis states) that amplifies exactly the marked
    divisor states computed above. The oracle is built by classically
    identifying which basis strings satisfy the property (computed by the
    classical trial-division check above, not asserted) and phase-flipping
    exactly those computational-basis states with multi-controlled-Z gates.
    This is run on the ideal AerSimulator (statevector-based, no noise).

Pass criterion:
    After running the circuit, the top measured outcomes (by shot count)
    must be exactly the marked divisor set computed classically above.
"""

import math
from itertools import product

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_divisors(n: int, lo: int, hi: int) -> list[int]:
    """Return {x in [lo, hi] : x > 1 and n % x == 0}, by trial division."""
    result = []
    for x in range(lo, hi + 1):
        if x > 1 and n % x == 0:
            result.append(x)
    return result


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase-flip each marked computational basis state (multi-controlled Z)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        # Flip zero-bits to 1 so the target pattern becomes all-ones,
        # apply a multi-controlled Z (via H-MCX-H on the last qubit), then
        # flip back.
        for i in zero_positions:
            qc.x(i)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.h(num_qubits - 1)
            qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
            qc.h(num_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.h(num_qubits - 1)
        qc.append(MCXGate(num_qubits - 1), list(range(num_qubits - 1)) + [num_qubits - 1])
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def main() -> None:
    N = 12
    num_qubits = 4
    lo, hi = 0, (2 ** num_qubits) - 1

    marked = classical_divisors(N, lo, hi)
    assert marked == [2, 3, 4, 6, 12], f"unexpected classical divisor set: {marked}"
    print(f"Classical property: divisors of {N} in [{lo},{hi}] greater than 1")
    print(f"Classical answer (marked set): {marked}")

    M = len(marked)
    search_space = 2 ** num_qubits
    # Optimal number of Grover iterations: floor(pi/4 * sqrt(N/M))
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / M)))
    print(f"Search space size = {search_space}, |marked| = {M}, Grover iterations = {iterations}")

    oracle = build_oracle(num_qubits, marked)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(num_qubits))
        qc.append(diffuser.to_instruction(), range(num_qubits))
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 8192
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (Qiskit prints classical bits MSB..LSB of creg,
    # matching qubit order 0..n-1 reversed) back to integers matching our
    # little-endian qubit encoding used in build_oracle.
    def bitstring_to_int(bs: str) -> int:
        return int(bs[::-1], 2)

    measured = sorted(
        ((bitstring_to_int(bs), c) for bs, c in counts.items()),
        key=lambda t: -t[1],
    )
    print("Top measured outcomes (value: count):")
    for val, c in measured[:M + 3]:
        print(f"  {val}: {c}")

    top_m_values = sorted(v for v, _ in measured[:M])
    expected = sorted(marked)

    print(f"Top-{M} measured values: {top_m_values}")
    print(f"Classical marked set:    {expected}")

    ok = top_m_values == expected
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
