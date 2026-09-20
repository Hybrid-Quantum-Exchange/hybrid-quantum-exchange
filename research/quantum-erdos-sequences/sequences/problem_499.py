"""
Erdos problem #499 — quantum-testable sequence attempt.

Source metadata (erdosproblems.com data, data/problems.yaml, entry "number: 499"):
    prize: "no"
    informal_status: proved (2025-11-29)
    formal_status: Lean (2025-11-29)
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION (read before trusting the PASS below):
    Problem #499 carries no OEIS sequence id in the source data (oeis: ["N/A"]).
    There is therefore no actual integer sequence from this problem to build a
    finite computable membership/divisibility/counting property around, and no
    honest way to construct a quantum circuit that verifies a term of "the
    Erdos #499 sequence" — there is no such sequence in the source data to
    verify against. Fabricating an OEIS id or a property not actually tied to
    this problem would violate the task's own instruction not to fake content.

    Per the task's fallback instructions, this script still contains a real,
    runnable Qiskit circuit and a real classical computation, but the property
    it tests is a generic small combinatorial search (Grover's algorithm
    finding the unique 3-bit value x such that x*x mod 8 == 1, i.e. searching
    {0,...,7} for square roots of 1 mod 8) rather than anything derived from
    Erdos problem #499's own (nonexistent, in this dataset) sequence. This is
    disclosed honestly: the quantum result is verified against a classical
    computation, but it is NOT a verification of any Erdos-499 sequence term,
    because no such OEIS sequence exists in the source data used.

Classical property actually computed and verified here:
    N = 16 (4 qubits). Find all x in {0,...,15} with x^2 mod 16 == 1.
    Classically checking every x: only x=1, 7, 9, 15 satisfy x^2 mod 16 == 1
    (1^2=1, 7^2=49=3*16+1, 9^2=81=5*16+1, 15^2=225=14*16+1); every other
    residue mod 16 fails. So the marked set is {1,7,9,15} (4 of 16 values,
    i.e. the units of (Z/16Z)* whose square is 1). This is a 25%-marked
    instance, for which Grover's algorithm with one iteration (the optimal
    iteration count for M=4, N=16) amplifies the marked amplitudes; with
    M = N/2 (as in an earlier 3-qubit attempt) the average amplitude is
    exactly zero and the Grover diffuser has no effect at all, which is why
    this script uses N=16 instead.
"""

import sys
from itertools import product

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def classical_marked_set(n_bits: int) -> set:
    """Classically compute {x in [0, 2^n_bits) : x^2 mod 2^n_bits == 1}."""
    modulus = 2 ** n_bits
    return {x for x in range(modulus) if (x * x) % modulus == 1}


def build_oracle(n_bits: int, marked: set) -> QuantumCircuit:
    """Phase-flip oracle marking each x in `marked` via multi-controlled Z,
    built from first principles with X-gates to remap 0-bits to controls."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for x in marked:
        bits = [(x >> i) & 1 for i in range(n_bits)]
        flip_qubits = [i for i, b in enumerate(bits) if b == 0]
        for q in flip_qubits:
            qc.x(q)
        # multi-controlled Z across all n_bits qubits, target = last qubit
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.mcx(list(range(n_bits - 1)), n_bits - 1)
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def grover_circuit(n_bits: int, marked: set, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))
    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_bits), range(n_bits))
    return qc


def main() -> bool:
    n_bits = 4  # N = 16
    marked = classical_marked_set(n_bits)
    expected = {1, 7, 9, 15}

    print(f"Classical marked set (x^2 mod 16 == 1) for x in 0..15: {sorted(marked)}")
    if marked != expected:
        print(f"FAIL: classical computation {sorted(marked)} != expected {sorted(expected)}")
        return False

    # Optimal iteration count for M=4 marked out of N=8: theta = asin(sqrt(M/N)),
    # iterations = round(pi/(4*theta) - 0.5)
    m, n = len(marked), 2 ** n_bits
    theta = np.arcsin(np.sqrt(m / n))
    iterations = max(1, round(np.pi / (4 * theta) - 0.5))
    print(f"Using {iterations} Grover iteration(s) for M={m}, N={n}")

    qc = grover_circuit(n_bits, marked, iterations)

    sim = AerSimulator()
    shots = 4096
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings c2c1c0 (qubit order little-endian per Qiskit)
    measured_values = {}
    for bitstring, count in counts.items():
        x = int(bitstring, 2)
        measured_values[x] = measured_values.get(x, 0) + count

    total_marked_counts = sum(measured_values.get(x, 0) for x in marked)
    success_rate = total_marked_counts / shots
    most_common = max(measured_values.items(), key=lambda kv: kv[1])[0]

    print(f"Measurement histogram (value: count): {sorted(measured_values.items())}")
    print(f"Most frequent measured value: {most_common}")
    print(f"Fraction of shots landing on a marked value {sorted(marked)}: {success_rate:.3f}")

    # Baseline (no amplification) success rate would be m/n; require the
    # quantum circuit to clear that baseline by a wide margin.
    baseline = m / n
    quantum_ok = (most_common in marked) and (success_rate > 3 * baseline)

    verified = quantum_ok and (marked == expected)
    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
