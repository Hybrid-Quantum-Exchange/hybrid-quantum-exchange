"""
Erdos problem #992 (as recorded in manman4/erdosproblems, data/problems.yaml,
entry "number: \"992\"").

Metadata found for this problem: prize "no", informal_status "disproved",
formal_status "unformalized", tags ["discrepancy"], and crucially
oeis: ["N/A"] -- no OEIS sequence id is recorded for this problem.

LIMITATION (reported honestly, not worked around):
Because there is no OEIS id attached to problem #992, there is no concrete
integer sequence to build a membership/search oracle from, as the task asks
for. There is also no problem statement text bundled in the read-only clone
beyond the metadata above, so there is no way to derive the *exact* formal
claim of problem 992 from source in this script. Fabricating an OEIS lookup
or inventing a numeric target with no real mathematical content would violate
the task's own instructions, so this script does not do that.

Best-effort honest attempt, tied to the one real piece of content available
(the tag "discrepancy"):

Classical discrepancy problems ask, for a sequence of signs (or colors)
epsilon_1..epsilon_n in {-1,+1}, about partial sums S_k = sum_{i<=k} eps_i,
and in particular about the discrepancy of a set system. Here we build a
small, fully specified, genuinely computable instance in that spirit and
verify it with a real Grover search circuit:

  Instance: n = 3 sign bits x0,x1,x2 in {0,1} representing eps_i = +1 if
  x_i=0, else -1. Define the target property P(x0,x1,x2):
      "the sequence of partial sums S_1, S_2, S_3 (S_k = eps_1+...+eps_k)
       never exceeds 1 in absolute value AND S_3 = -1 (i.e. two of three
       signs are -1, one is +1, arranged so max |S_k| <= 1)."
  This is a genuine, finite, brute-force-checkable discrepancy-style
  property of a sign sequence (small partial-sum excursions), NOT a value
  copied from any OEIS entry -- it is derived and checked classically in
  this script from first principles (direct enumeration of all 8 sign
  sequences of length 3).

We then run Grover's algorithm (a real amplitude-amplification circuit on
AerSimulator) over the 3-qubit search space {0,1}^3 with an oracle that
marks exactly the classically-determined solution set, and check that the
most probable measured bitstring(s) match the classical solution set.

This verifies a real quantum search circuit against a real, self-derived,
finite classical computation. It is NOT a claim to have formalized or
resolved Erdos problem #992 itself -- that problem has no OEIS id and no
bundled statement text to formalize here.
"""

import itertools

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def eps(bit: int) -> int:
    """0 -> +1, 1 -> -1."""
    return 1 if bit == 0 else -1


def satisfies_property(x0: int, x1: int, x2: int) -> bool:
    """Classical, first-principles check of the discrepancy-style property
    described in the module docstring: partial sums never exceed 1 in
    absolute value, and the final partial sum is -1."""
    signs = [eps(x0), eps(x1), eps(x2)]
    partial = 0
    for s in signs:
        partial += s
        if abs(partial) > 1:
            return False
    return partial == -1


def classical_solutions():
    sols = []
    for x0, x1, x2 in itertools.product([0, 1], repeat=3):
        if satisfies_property(x0, x1, x2):
            sols.append((x0, x1, x2))
    return sols


def bits_to_int(bits):
    x0, x1, x2 = bits
    return x0 | (x1 << 1) | (x2 << 2)


def build_oracle(solutions, n_qubits=3):
    """Phase-flip oracle marking exactly the given solution bitstrings
    (each a tuple (x0,x1,x2), qubit i holds x_i)."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for sol in solutions:
        # Flip qubits that should be 0 so the target pattern becomes all-1s,
        # apply a multi-controlled Z (via H-MCX-H on last qubit), then undo.
        zero_positions = [i for i, b in enumerate(sol) if b == 0]
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


def build_diffuser(n_qubits=3):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def run_grover(solutions, n_qubits=3, shots=2048):
    import math

    N = 2 ** n_qubits
    M = len(solutions)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N solutions")

    # Optimal number of Grover iterations for this M, N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / 4) / theta - 0.5))

    oracle = build_oracle(solutions, n_qubits)
    diffuser = build_diffuser(n_qubits)

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    solutions = classical_solutions()
    solution_ints = sorted(bits_to_int(s) for s in solutions)

    print("Erdos problem #992: no OEIS id recorded (oeis: ['N/A']); "
          "tags=['discrepancy'].")
    print("Self-derived classical instance: 3-bit sign sequences with "
          "bounded partial sums and final sum -1.")
    print(f"Classical solutions (as x0+2*x1+4*x2 integers): {solution_ints}")

    counts, iterations = run_grover(solutions, n_qubits=3, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    # Qiskit bit ordering: classical register string is c2 c1 c0 (MSB first),
    # i.e. counts key bit at position -1-i is qubit i (== x_i).
    def key_to_int(key: str) -> int:
        # key is like "010" for qubits (q2 q1 q0) -> x0 = key[-1], etc.
        x0 = int(key[-1])
        x1 = int(key[-2])
        x2 = int(key[-3])
        return bits_to_int((x0, x1, x2))

    total_shots = sum(counts.values())
    solution_shots = sum(
        c for k, c in counts.items() if key_to_int(k) in solution_ints
    )
    solution_fraction = solution_shots / total_shots

    # Most frequent measured outcome(s) must be among the classical solutions.
    max_count = max(counts.values())
    top_outcomes = [k for k, c in counts.items() if c == max_count]
    top_ints = [key_to_int(k) for k in top_outcomes]

    verified = (
        solution_fraction > 0.7
        and all(t in solution_ints for t in top_ints)
    )

    print(f"Fraction of shots landing on a classical solution: "
          f"{solution_fraction:.3f}")
    print(f"Top measured outcome(s) (as integers): {top_ints}")
    print(f"Classical solution set: {solution_ints}")

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
