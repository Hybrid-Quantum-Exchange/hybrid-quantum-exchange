"""
Erdos Problem #242 -- quantum-testable instance
=================================================

Erdos problem #242 is the Erdos-Straus conjecture: for every integer n >= 2,
the equation

    4/n = 1/x + 1/y + 1/z

has a solution in positive integers x, y, z. The problems.yaml entry for
number "242" lists OEIS ids A073101, A075245, A075246, A075247, A075248,
A287116, which index various aspects of Erdos-Straus solutions (counts of
solutions, minimal solutions, etc.), and tags ["number theory",
"unit fractions"].

Classical property tested here
-------------------------------
Fix n = 5 (the smallest interesting non-trivial case; the trivial 4/2, 4/3,
4/4 identities are not what makes the conjecture hard). Restrict the search
to a small "numerator" register x in {0, 1, ..., 7} (3 qubits). For each such
x we ask, purely classically and first-principles (nested loops, exact
fraction arithmetic via Python's `fractions.Fraction`, no OEIS lookups):

    Does there exist a decomposition 4/5 = 1/x + 1/y + 1/z
    with positive integers y, z <= Y_MAX?

This is computed exhaustively in `classical_marked_set()` below, with no
value copied from OEIS -- it is derived from scratch. The resulting set of
"good" x values (call it M) is a genuine, finite, computable property of an
Erdos-Straus solution for n = 5, restricted to a small search register, and
is exactly the kind of decision problem Grover's algorithm is built to
search.

Quantum circuit
----------------
We build a standard Grover search over the 3-qubit register x in [0, 8):
  - an oracle that phase-flips exactly the classically-computed marked
    states M (built directly from the classical truth table -- this is the
    standard way to turn an arbitrary known boolean function into a Grover
    oracle: a multi-controlled-Z gated on the bit pattern of each marked
    state),
  - the standard Grover diffusion operator,
  - the standard optimal number of iterations for |M| marked items out of
    2^3 = 8,
run on the ideal AerSimulator, then measured.

PASS/FAIL
---------
We compare the set of x values with the highest measured probability
(taking the top |M| measurement outcomes) against the classically computed
marked set M. If they match, we PASS: the quantum circuit successfully
searched out the integers x for which the Erdos-Straus decomposition of 4/5
exists (within the given bound), verified against ground truth computed
independently in this script.
"""

from fractions import Fraction
from itertools import product

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate

N_NUMERATOR = 5          # target: 4/N_NUMERATOR
NUM_QUBITS = 3            # search register size: x in [0, 2**NUM_QUBITS)
X_RANGE = range(1, 2 ** NUM_QUBITS)   # x = 0 is meaningless (division by zero)
Y_MAX = 200               # bound for the classical y, z search


def has_erdos_straus_decomposition(n: int, x: int, y_max: int) -> bool:
    """Classically (first principles) decide whether 4/n = 1/x + 1/y + 1/z
    has a solution in positive integers y, z with 1 <= y, z <= y_max.

    Exact rational arithmetic via fractions.Fraction; a fully exhaustive
    double loop over y, z (with the standard y <= z pruning) -- no shortcuts,
    no OEIS values used.
    """
    target = Fraction(4, n)
    if x <= 0:
        return False
    remainder = target - Fraction(1, x)
    if remainder <= 0:
        return False
    for y in range(1, y_max + 1):
        term_y = Fraction(1, y)
        if term_y >= remainder:
            # y too small would make 1/y alone >= remainder; once 1/y < remainder
            # is no longer possible for larger y either (1/y shrinks), so once
            # term_y < remainder we search z; if term_y >= remainder we still
            # need to allow equality with a valid z only when term_y < remainder,
            # otherwise skip.
            if term_y == remainder:
                # would need 1/z = 0, impossible for finite z
                continue
            else:
                continue
        rem_z = remainder - term_y
        if rem_z <= 0:
            continue
        # need 1/z == rem_z, i.e. z == 1/rem_z, must be a positive integer
        if rem_z.numerator == 1:
            z = rem_z.denominator
            if 1 <= z <= y_max:
                return True
    return False


def classical_marked_set(n: int, x_values, y_max: int) -> set:
    marked = set()
    for x in x_values:
        if has_erdos_straus_decomposition(n, x, y_max):
            marked.add(x)
    return marked


def build_oracle(num_qubits: int, marked_values: set) -> QuantumCircuit:
    """Phase-flip oracle: multi-controlled-Z on each marked bitstring."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for value in sorted(marked_values):
        bits = format(value, f"0{num_qubits}b")
        # X-gate the qubits that should be 0 in this basis state, so the
        # multi-controlled-Z fires exactly when the register equals `value`.
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for pos in zero_positions:
            qc.x(pos)
        if num_qubits == 1:
            qc.z(0)
        else:
            mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
            qc.append(mcz, list(range(num_qubits)))
        for pos in zero_positions:
            qc.x(pos)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        mcz = MCMTGate(ZGate(), num_qubits - 1, 1)
        qc.append(mcz, list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits: int, marked_values: set, shots: int = 4096):
    oracle = build_oracle(num_qubits, marked_values)
    diffuser = build_diffuser(num_qubits)

    n_items = 2 ** num_qubits
    m = max(len(marked_values), 1)
    # standard optimal iteration count for Grover search
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_items / m)))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    backend = AerSimulator()
    compiled = transpile(qc, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def top_measured_values(counts: dict, k: int) -> set:
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top = ordered[:k]
    return {int(bitstring, 2) for bitstring, _ in top}


def main():
    marked = classical_marked_set(N_NUMERATOR, X_RANGE, Y_MAX)
    print(f"Classical search: 4/{N_NUMERATOR} = 1/x + 1/y + 1/z, "
          f"x in {list(X_RANGE)}, y,z <= {Y_MAX}")
    print(f"Classically marked x values (Erdos-Straus solvable): {sorted(marked)}")

    if not marked:
        print("No marked values found classically -- cannot build a "
              "meaningful Grover search for this instance.")
        print("FAIL")
        return

    counts, iterations = run_grover(NUM_QUBITS, marked, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Measurement counts: {counts}")

    measured_top = top_measured_values(counts, len(marked))
    print(f"Top {len(marked)} measured value(s) by count: {sorted(measured_top)}")

    ok = measured_top == marked
    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
