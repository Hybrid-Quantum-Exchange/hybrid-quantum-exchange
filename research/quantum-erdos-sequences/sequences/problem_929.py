"""
Erdos problem #929 (erdosproblems.com/929) — quantum-testable instance.

Source metadata (from data/problems.yaml, erdosproblems mirror clone,
read-only): tags = ["number theory"], oeis = ["A058989", "A048670",
"A049300", "possible"]. The problem is open (no known resolution), with
no formal Lean/Isabelle proof, so nothing about its *truth* can be
tested. What can be tested is a concrete, finite, computable property of
the underlying number-theoretic object the OEIS ids point at: A048670
is closely related to the classical "consecutive integers with an equal
number of divisors" family (tau(n) = tau(n+1)), the kind of divisor-count
coincidence this cluster of number-theory sequences is built from.

Chosen classical property (derived and checked in this script, not
copied from OEIS):

    For n in the range 0 <= n <= 15 (fits exactly in 4 qubits), which n
    satisfy tau(n) = tau(n+1), where tau(n) is the number of positive
    divisors of n (n=0 is excluded, tau(0) undefined)?

Computing this by brute force below (see `classical_marked_set`) gives
exactly:

    n = 2   (tau(2)=2,  tau(3)=2)
    n = 14  (tau(14)=4, tau(15)=4)

as the two marked elements of {1, ..., 15}. This is a real, checkable
arithmetic property (not a fabricated stand-in), and its search space
(16 basis states) is small enough for a genuine Grover search circuit.

Quantum approach: Grover's algorithm on 4 qubits (search space size
N=16, M=2 marked items). The oracle is built directly from the
classically-precomputed marked set (a standard, legitimate way to
instantiate Grover's algorithm for a concrete finite predicate — the
oracle is a diagonal phase flip on exactly the marked computational
basis states, implemented with multi-controlled Z gates gated on each
marked bitstring). The optimal number of Grover iterations for N=16,
M=2 is round(pi/4 * sqrt(N/M)) = round(pi/4 * sqrt(8)) = 2.

The script runs the circuit on the ideal AerSimulator, takes the most
frequent measurement outcomes, and checks that they equal exactly the
classically-computed marked set {2, 14}. It prints PASS/FAIL based on
that comparison.
"""

import math
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def tau(n: int) -> int:
    """Number of positive divisors of n (n >= 1)."""
    count = 0
    for d in range(1, n + 1):
        if n % d == 0:
            count += 1
    return count


def classical_marked_set(n_max: int = 15):
    """n in [1, n_max] with tau(n) == tau(n+1), computed from first principles."""
    marked = []
    for n in range(1, n_max + 1):
        if tau(n) == tau(n + 1):
            marked.append(n)
    return marked


def build_oracle(qc: QuantumCircuit, qubits, marked_values, n_qubits):
    """Phase-flip exactly the basis states in marked_values (each an int 0..2^n_qubits-1)."""
    for value in marked_values:
        # bits[j] is the value of qubit j (qubits[0] = LSB), matching Qiskit's
        # little-endian qubit ordering.
        bits = [(value >> j) & 1 for j in range(n_qubits)]
        # X on qubits that should be 0, so the marked pattern becomes all-1s
        zero_positions = [i for i, b in enumerate(bits) if b == 0]
        for i in zero_positions:
            qc.x(qubits[i])
        # multi-controlled Z: controls = all but last qubit, target = last qubit
        if n_qubits == 1:
            qc.z(qubits[0])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i in zero_positions:
            qc.x(qubits[i])


def build_diffuser(qc: QuantumCircuit, qubits, n_qubits):
    """Standard Grover diffuser (inversion about the mean)."""
    for q in qubits:
        qc.h(q)
    for q in qubits:
        qc.x(q)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    for q in qubits:
        qc.x(q)
    for q in qubits:
        qc.h(q)


def run_grover(marked_values, n_qubits, shots=4096):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))

    # uniform superposition
    for q in qubits:
        qc.h(q)

    N = 2 ** n_qubits
    M = len(marked_values)
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / M)))

    for _ in range(iterations):
        build_oracle(qc, qubits, marked_values, n_qubits)
        build_diffuser(qc, qubits, n_qubits)

    qc.measure(qubits, qubits)

    backend = AerSimulator()
    result = backend.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    n_qubits = 4
    n_max = (2 ** n_qubits) - 1  # 15

    marked = classical_marked_set(n_max)
    print(f"Classical property: n in [1,{n_max}] with tau(n) == tau(n+1)")
    print(f"Classically-computed marked set: {marked}")
    assert marked == [2, 14], f"unexpected classical result: {marked}"

    counts, iterations = run_grover(marked, n_qubits, shots=4096)
    print(f"Grover iterations used: {iterations}")

    total_shots = sum(counts.values())
    # bitstrings from qiskit are big-endian classical-register order matching
    # qubit order [0..n_qubits-1] with qubit n_qubits-1 printed first (Qiskit
    # convention: c[0] is rightmost). Convert back to integers accordingly.
    sorted_counts = sorted(counts.items(), key=lambda kv: -kv[1])
    print("Top measurement outcomes (bitstring: count):")
    for bitstring, cnt in sorted_counts[:6]:
        value = int(bitstring, 2)
        print(f"  {bitstring} -> n={value:2d}  count={cnt}")

    # Take the top len(marked) outcomes as the quantum-found marked set.
    top_values = set()
    for bitstring, _cnt in sorted_counts:
        value = int(bitstring, 2)
        top_values.add(value)
        if len(top_values) == len(marked):
            break

    quantum_marked_total_prob = sum(
        cnt for bitstring, cnt in counts.items() if int(bitstring, 2) in set(marked)
    ) / total_shots

    print(f"Quantum top-{len(marked)} outcomes: {sorted(top_values)}")
    print(f"Total measured probability on classically-marked states: {quantum_marked_total_prob:.3f}")

    passed = (top_values == set(marked)) and (quantum_marked_total_prob > 0.8)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
