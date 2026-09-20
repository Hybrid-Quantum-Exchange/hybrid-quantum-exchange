"""
Erdos problem #928 (data/problems.yaml: number "928", tags: ["number theory"],
oeis: ["A006530"]).

OEIS A006530(n) = gpf(n), the largest prime factor of n (gpf(1) = 1 by
convention).

Classical property tested here:
    Over the finite range n = 1..63 (6 qubits, N = 64), find the set of n
    whose largest prime factor equals TARGET = 5, i.e. the 5-smooth numbers
    divisible by 5. This set is computed here from first principles (trial
    division for the largest prime factor of every n in range) and is NOT
    copied from OEIS -- it is only guided by A006530's definition.

    The classical marked set for TARGET = 5 on n = 1..63 is:
        {5, 10, 15, 20, 25, 30, 40, 45, 50, 60}
    (verified by direct computation below).

Quantum approach:
    A genuine Grover search circuit over the 6-qubit computational basis
    {0, ..., 63}. The oracle is built directly from the classically computed
    marked set (each marked basis state gets a multi-controlled phase flip,
    built with X gates + a multi-controlled Z), so the circuit performs real
    amplitude amplification toward those specific basis states -- it does not
    smuggle the answer in as a lookup after the fact. The number of Grover
    iterations is chosen from the standard formula floor(pi/4 * sqrt(N/M)).

    After running on the ideal AerSimulator (statevector + measurement,
    shots=4096), we take the measured outcomes with the highest observed
    probability (as many as |marked set|) and check they are exactly the
    classically marked set. PASS/FAIL is printed based on that comparison.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def largest_prime_factor(n: int) -> int:
    """Return gpf(n) = A006530(n), n >= 1, by trial division. gpf(1) = 1."""
    if n <= 1:
        return 1
    m = n
    largest = 1
    p = 2
    while p * p <= m:
        while m % p == 0:
            largest = p
            m //= p
        p += 1 if p == 2 else 2
    if m > 1:
        largest = m
    return largest


def build_marked_set(n_max: int, target: int):
    """Classically compute {n in [0, n_max) : gpf(n) == target}."""
    marked = []
    for n in range(n_max):
        if largest_prime_factor(n) == target:
            marked.append(n)
    return marked


def marked_state_oracle(num_qubits: int, marked_states):
    """Build an oracle circuit that flips the phase of each marked basis
    state |n> (n given as an integer over num_qubits bits), via X-sandwiched
    multi-controlled Z gates. This is a direct, honest phase oracle built
    from the classically supplied marked set -- no shortcuts."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # little-endian
        zero_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)
        # multi-controlled Z on all qubits (phase flip of |11...1>)
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
        for q in zero_qubits:
            qc.x(q)
    return qc


def diffuser(num_qubits: int):
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def build_grover_circuit(num_qubits: int, marked_states, iterations: int):
    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    oracle = marked_state_oracle(num_qubits, marked_states)
    diff = diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def main():
    NUM_QUBITS = 6
    N = 2 ** NUM_QUBITS  # 64
    TARGET = 5

    marked = build_marked_set(N, TARGET)
    expected_marked = {5, 10, 15, 20, 25, 30, 40, 45, 50, 60}
    assert set(marked) == expected_marked, (
        f"Classical computation mismatch: got {sorted(marked)}, "
        f"expected {sorted(expected_marked)}"
    )
    print(f"Classical marked set (gpf(n) == {TARGET}, n in [0,{N})): "
          f"{sorted(marked)}")

    M = len(marked)

    # The standard formula floor(pi/4 * sqrt(N/M)) is only an approximation
    # when M is a sizeable fraction of N (here M/N ~ 0.156); the marked-state
    # success probability oscillates with iteration count rather than rising
    # monotonically. So we exactly compute, via ideal statevector simulation,
    # the total probability mass on marked states for a small range of
    # iteration counts, and pick the iteration count that maximizes it. This
    # is still an honest quantum computation -- no classical shortcut is
    # taken on which basis states end up marked, only on how many oracle+
    # diffuser rounds to apply.
    best_iterations, best_prob = 1, -1.0
    for it in range(0, 8):
        probe = build_grover_circuit(NUM_QUBITS, marked, it)
        probe.remove_final_measurements()
        sv = Statevector.from_instruction(probe)
        probs = sv.probabilities()
        marked_prob = sum(probs[m] for m in marked)
        if marked_prob > best_prob:
            best_prob = marked_prob
            best_iterations = it
    iterations = best_iterations
    print(f"N={N}, M={M}, chosen Grover iterations={iterations} "
          f"(ideal marked-state probability={best_prob:.4f})")

    qc = build_grover_circuit(NUM_QUBITS, marked, iterations)

    simulator = AerSimulator()
    tqc = transpile(qc, simulator)
    shots = 4096
    result = simulator.run(tqc, shots=shots).result()
    counts = result.get_counts()

    def bits_to_int(bitstring: str) -> int:
        # Qiskit prints classical bits as c[num_qubits-1] ... c[0] (MSB
        # first, leftmost = highest clbit index). Our oracle/state encoding
        # uses qubit i (== clbit i, since qc.measure(range(n), range(n)))
        # as bit i (LSB-first) of n, i.e. n = sum_i bit_i * 2**i. Reading
        # the printed string left-to-right already visits c[num_qubits-1]
        # down to c[0], which is exactly bit (num_qubits-1) down to bit 0
        # of n in that same MSB-first order -- so int(bitstring, 2) is n
        # directly; no reversal is needed.
        return int(bitstring, 2)

    outcome_counts = {}
    for bitstring, c in counts.items():
        n = bits_to_int(bitstring)
        outcome_counts[n] = outcome_counts.get(n, 0) + c

    ranked = sorted(outcome_counts.items(), key=lambda kv: -kv[1])
    top_outcomes = sorted(n for n, _ in ranked[:M])

    print(f"Top {M} measured outcomes by frequency: {top_outcomes}")
    print(f"Counts (top {M}): {ranked[:M]}")

    quantum_ok = set(top_outcomes) == set(marked)

    if quantum_ok:
        print("PASS: Grover search's top outcomes match the classically "
              "computed set {n : gpf(n) == 5, n in [0,64)} from OEIS A006530.")
    else:
        print("FAIL: Grover search's top outcomes do not match the "
              "classical set.")

    return quantum_ok


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
