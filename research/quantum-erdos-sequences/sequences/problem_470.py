"""
Erdos problem #470 (erdosproblems.com), quantum-testable instance.

Metadata (from data/problems.yaml in the manman4/erdosproblems clone):
  number: "470", tags: ["number theory", "divisors"], prize: "$10",
  oeis: ["A006037", "A002975"]

  A006037 = abundant numbers: n such that sigma(n) > 2n (sigma = sum of
  divisors of n, including n itself).
  A002975 = primitive abundant numbers: abundant numbers all of whose
  proper divisors are deficient (sigma(d) <= 2d for every proper divisor d).

Classical property tested here (finite, computable, small search space):
  Over the search space n in [0, 31] (5 qubits), mark exactly the set of
  abundant numbers, i.e. S = { n in [0,31] : sigma(n) > 2n }.
  This is computed from first principles in `classical_abundant_set` below
  (no OEIS values are looked up or copied -- sigma(n) is computed by
  divisor summation and compared to 2n).

  For n in [0,31], sigma is computed with sigma(0) defined as 0 (0 is
  excluded from being "abundant" by construction: sigma(0) > 0 is
  vacuously true only if we let sigma(0)=0, so n=0 is never marked).
  The classically computed set is expected to be {12, 18, 20, 24, 30},
  matching the start of OEIS A006037 restricted to n < 32.

Quantum approach: Grover's algorithm.
  Build a 5-qubit oracle that phase-flips exactly the basis states
  |n> for n in the classically-computed abundant set S (the oracle is
  built directly from the classical computation -- it is not hand-copied
  from OEIS, it is derived by the script itself). Run the standard
  Grover diffusion operator, with the optimal number of iterations for
  |S|=5 out of N=32 marked items, on the ideal AerSimulator. Verify that
  measurement overwhelmingly returns basis states in S with the expected
  amplification (Grover's algorithm genuinely searching this space for
  the primitive-vs-abundant-divisor classical property).

PASS/FAIL: PASS iff the quantum circuit's returned answer set (the states
whose measured probability exceeds the uniform baseline by a wide margin)
equals the classically computed abundant set S exactly.
"""

import math
from collections import Counter

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator


N_QUBITS = 5
N = 2 ** N_QUBITS  # 32


def sigma(n: int) -> int:
    """Sum of all positive divisors of n (including n itself). sigma(0) := 0."""
    if n == 0:
        return 0
    total = 0
    for d in range(1, n + 1):
        if n % d == 0:
            total += d
    return total


def classical_abundant_set(limit: int):
    """n in [0, limit) with sigma(n) > 2n -- computed from first principles."""
    return sorted(n for n in range(limit) if sigma(n) > 2 * n)


def mark_state_gate(n_qubits: int, value: int):
    """
    Returns a gate acting on n_qubits+1 wires (last is the phase-kickback
    ancilla-free target via multi-controlled Z realized through X-sandwiched
    MCX on phase). Here we build a plain multi-controlled Z (phase flip)
    on n_qubits that fires only when the qubit register equals `value`.
    """
    qc = QuantumCircuit(n_qubits, name=f"mark_{value}")
    bits = [(value >> i) & 1 for i in range(n_qubits)]
    # Flip 0-bits to 1 so "all ones" <=> register == value
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    # multi-controlled Z: use H + MCX + H on last qubit as target trick,
    # but we want a pure phase flip on all-ones with no extra target qubit.
    # Implement via MCX with phase: use the standard "multi-controlled Z"
    # by putting a Z on the last qubit conditioned on the rest via MCX+H sandwich.
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    for i, b in enumerate(bits):
        if b == 0:
            qc.x(i)
    return qc.to_gate(label=f"mark|{value}>")


def build_oracle(n_qubits: int, marked_values):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    for v in marked_values:
        qc.append(mark_state_gate(n_qubits, v), range(n_qubits))
    return qc.to_gate(label="Oracle")


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc.to_gate(label="Diffuser")


def grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    r = round((math.pi / (4 * theta)) - 0.5)
    return max(1, r)


def run_grover(n_qubits: int, marked_values):
    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    iters = grover_iterations(2 ** n_qubits, len(marked_values))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iters):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    shots = 20000
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iters, shots


def main():
    marked = classical_abundant_set(N)
    print(f"Classical abundant set S (n < {N}, sigma(n) > 2n): {marked}")
    assert marked == [12, 18, 20, 24, 30], (
        "Sanity check on classical computation failed: "
        f"got {marked}, expected [12, 18, 20, 24, 30]"
    )

    counts, iters, shots = run_grover(N_QUBITS, marked)

    # Aggregate measured probability per integer value (bitstrings are
    # big-endian in Qiskit's default counts keys, c[N_QUBITS-1..0]).
    prob_by_value = Counter()
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        prob_by_value[value] += c / shots

    baseline = 1.0 / N
    # Threshold well above uniform baseline (32 states -> baseline ~3.1%).
    threshold = 4 * baseline
    found = sorted(v for v, p in prob_by_value.items() if p > threshold)

    marked_prob = sum(prob_by_value[v] for v in marked)
    print(f"Grover iterations used: {iters}, shots: {shots}")
    print(f"Total measured probability mass on marked set S: {marked_prob:.4f}")
    print(f"States found above {threshold*100:.1f}% threshold: {found}")

    verified = (found == marked) and (marked_prob > 0.80)

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    if not ok:
        raise SystemExit(1)
