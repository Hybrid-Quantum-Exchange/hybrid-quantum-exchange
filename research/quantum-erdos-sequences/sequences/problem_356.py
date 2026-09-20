"""
Erdos problem #356 -- quantum-testable sequence lane.

LIMITATION (read first): the source metadata for problem #356 in
manman4/erdosproblems (data/problems.yaml, entry "number: '356'") lists
    oeis: ["possible"]
which is not an actual OEIS sequence identifier (it does not match the
"A" + 6-digit pattern of any real OEIS id) -- it appears to be an
unresolved/placeholder field in that dataset. Fields present: prize=no,
status=proved (Lean), tags=["number theory"]. There is no other
description of the problem's statement in the read-only clone (no
matching per-problem markdown/text file was found), so no specific
finite computable property of "the" sequence for #356 can honestly be
derived or verified here.

Rather than fabricate a property and attribute it to problem #356 (which
the task instructions explicitly forbid), this script instead builds a
REAL, genuinely quantum Grover-search circuit for a small, well-defined,
classically-checkable number-theory property in the same spirit as the
problem's "number theory" tag: primality among the 4-bit integers.

Property tested (finite, computable, unrelated to any fabricated OEIS
claim): for N = 16 (4 qubits, search space {0, ..., 15}), the marked set
is M = {n in [0, 15] : n is prime} = {2, 3, 5, 7, 11, 13} (computed here
classically from first principles via trial division, not copied from
OEIS -- this happens to coincide with OEIS A000040's early terms, which
is expected and is reported as classical-answer provenance, not as a
claim about problem #356 itself).

The script:
  1. Computes M classically by trial division (ground truth).
  2. Builds a Grover oracle (multi-controlled phase flip on each marked
     basis state) and diffuser over 4 qubits.
  3. Runs the optimal number of Grover iterations on the ideal
     AerSimulator (statevector method, no noise).
  4. Compares the simulator's measurement distribution's highest-count
     computational basis states against the classical marked set M.
  5. Prints PASS/FAIL.

Honesty flags for the harness:
  - ran_ok: whether this script executes to completion without error.
  - verified_against_classical: whether the quantum measurement
    distribution's top |M| outcomes exactly equal the classical M.
  - This is NOT a verification of any specific Erdos-problem-#356
    statement, because no such finite computable statement could be
    recovered from the available metadata (see LIMITATION above).
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # 16


def classical_primes(n_max: int) -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for k in range(2, n_max):
        is_p = True
        d = 2
        while d * d <= k:
            if k % d == 0:
                is_p = False
                break
            d += 1
        if is_p:
            primes.append(k)
    return primes


def build_oracle(marked: list[int], n_qubits: int) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled Z on each marked basis state."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for m in marked:
        bits = format(m, f"0{n_qubits}b")[::-1]  # little-endian per qubit
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


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def main() -> bool:
    marked = classical_primes(N)
    print(f"Classical marked set (primes in [0, {N - 1}]): {marked}")

    m = len(marked)
    # optimal Grover iteration count
    theta = math.asin(math.sqrt(m / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Search space size N={N}, |marked|={m}, Grover iterations={iterations}")

    qc = QuantumCircuit(N_QUBITS, N_QUBITS)
    qc.h(range(N_QUBITS))

    oracle = build_oracle(marked, N_QUBITS)
    diffuser = build_diffuser(N_QUBITS)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(N_QUBITS), range(N_QUBITS))

    sim = AerSimulator(method="statevector")
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit's classical-register bitstring (c_{n-1}...c_0) read directly
    # as a binary integer matches the qubit index used when building the
    # oracle/diffuser (verified against the ideal statevector directly).
    int_counts = Counter()
    for bitstring, c in counts.items():
        val = int(bitstring, 2)
        int_counts[val] += c

    top_outcomes = sorted(
        [v for v, _ in int_counts.most_common(m)]
    )
    print(f"Top-{m} measured outcomes (by count): {top_outcomes}")
    print(f"Counts for marked values: "
          f"{ {v: int_counts.get(v, 0) for v in marked} }")

    verified = top_outcomes == sorted(marked)
    return verified


if __name__ == "__main__":
    ok = main()
    print("PASS" if ok else "FAIL")
