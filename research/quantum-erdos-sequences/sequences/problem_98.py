"""
Erdos problem #98 -- quantum-testable sequence lane.

Source metadata (from erdosproblems.com data, data/problems.yaml, entry
"number: '98'"):
    prize: none
    tags: ["geometry", "distances"]
    oeis: ["possible"]

LIMITATION, stated honestly up front: the "oeis" field for problem #98 in the
upstream data file is the literal string "possible", not an actual OEIS
sequence identifier. There is no real A-number to derive a sequence from for
this problem, so this script cannot build a circuit that tests membership in,
or a term of, "the OEIS sequence for problem 98" -- there isn't one. Per the
task's fallback instruction, this is the best-effort honest substitute: a
genuine, self-contained finite/computable question drawn from the problem's
own subject area (tags: geometry, distances -- Erdos problem #98 concerns
integer distances between points in the plane, in the spirit of the
Erdos-Anning theorem), solved both classically and with a real Grover search
circuit on Qiskit's AerSimulator, with the two answers compared.

Classical property tested
--------------------------
Fix y = 3. Over the finite domain x in {0, 1, ..., 63} (6 qubits, N = 64,
satisfying the "N <= ~64" bound), define the property

    P(x)  <=>  x^2 + y^2 is a perfect square

i.e. (x, y) is the leg pair of a Pythagorean triple, equivalently the point
(x, y) is at an *integer* distance from the origin -- exactly the kind of
integer-distance question problem #98's tags point at. This is computed from
first principles in `classical_marked_set()` below (integer square-root
check, no OEIS lookup, no hard-coded literal answer).

For y = 3 over x in 0..63 this gives exactly one marked value: x = 4
(3-4-5 triangle: 4^2 + 3^2 = 25 = 5^2).

Quantum approach
-----------------
A real Grover search circuit (6 qubits over N = 2^6 = 64 basis states, one
marked state) is built on AerSimulator:
  - an oracle that phase-flips the computational basis states in the
    classically-computed marked set (built generically from that set, not by
    special-casing the single-marked-state case),
  - the standard Grover diffusion operator,
  - ceil(pi/4 * sqrt(N/|marked|)) iterations, computed from the actual size
    of the marked set.
The circuit is run, measurement counts collected, and the most frequent
outcome is compared against the classically-computed marked set to print
PASS or FAIL.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_QUBITS = 6
N = 2 ** N_QUBITS  # 64
Y_FIXED = 3


def classical_marked_set():
    """First-principles classical computation of P(x) for x in 0..N-1.

    P(x) <=> x^2 + Y_FIXED^2 is a perfect square.
    Returns a sorted list of marked integers.
    """
    marked = []
    for x in range(N):
        s = x * x + Y_FIXED * Y_FIXED
        r = math.isqrt(s)
        if r * r == s:
            marked.append(x)
    return marked


def build_oracle(marked, n_qubits):
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z: phase flip |111...1> using H-MCX-H on target
        qc.h(n_qubits - 1)
        if n_qubits - 1 == 0:
            qc.z(0)
        else:
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    if n_qubits - 1 == 0:
        qc.z(0)
    else:
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits):
    n_marked = len(marked)
    if n_marked == 0:
        raise ValueError("no marked states -- Grover search is undefined")
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / n_marked)))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc, iterations


def main():
    marked = classical_marked_set()
    print(f"Classical marked set for y={Y_FIXED}, x in 0..{N-1}: {marked}")
    expected_x = 4  # 3-4-5 triangle, cross-check
    assert expected_x in marked, "classical computation disagrees with known 3-4-5 triple"

    qc, iterations = build_grover_circuit(marked, N_QUBITS)
    print(f"Grover iterations used: {iterations}")

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=2048).result()
    counts = result.get_counts()

    # Qiskit bit order: rightmost char is qubit 0 -> integer value directly.
    def bitstring_to_int(bs):
        return int(bs, 2)

    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    top_bitstring, top_count = sorted_counts[0]
    top_value = bitstring_to_int(top_bitstring)

    total_shots = sum(counts.values())
    marked_shots = sum(c for bs, c in counts.items() if bitstring_to_int(bs) in marked)
    marked_fraction = marked_shots / total_shots

    print(f"Top measured value: {top_value} (count {top_count}/{total_shots})")
    print(f"Fraction of shots landing on a marked (classically verified) state: {marked_fraction:.3f}")

    quantum_found_correct = top_value in marked
    amplification_worked = marked_fraction > (len(marked) / N) * 3  # well above uniform baseline

    verified = quantum_found_correct and amplification_worked

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
