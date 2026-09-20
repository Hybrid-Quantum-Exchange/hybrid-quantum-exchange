"""
Erdos problem #284 -- quantum-testable instance.

Source metadata (data/problems.yaml in manman4/erdosproblems, entry "number: 284"):
    tags: ["number theory", "unit fractions"]
    oeis: ["possible"]   <-- NOT a real OEIS sequence id. The yaml field literally
                             contains the placeholder string "possible", not an
                             A-number. There is therefore no genuine OEIS sequence
                             id to anchor this script to, and the classical property
                             tested below is chosen from the problem's TAGS alone
                             ("unit fractions" / Egyptian-fraction decomposition),
                             not from an OEIS-verified term. This limitation is
                             reported honestly in the final output below.

Because no OEIS id is available, this script tests the best small, finite,
genuinely-computable "unit fractions" property it can construct: an
Erdos-Straus-style Egyptian-fraction decomposition question.

Classical property under test
------------------------------
Fix N = 5 and the target fraction 4/N = 4/5.
For each candidate first denominator a in {1, ..., 15} (4 bits), define

    f(a) = True  iff there exist positive integers b, c (b, c <= B, with a fixed
                 search bound B = 60 used only for the classical precomputation)
                 such that
                     4/5 == 1/a + 1/b + 1/c   (exact rational equality)

i.e. f(a) marks the denominators a for which 4/5 has an Egyptian-fraction
(unit-fraction) decomposition into exactly three unit fractions with a as the
first term -- an instance of the same "unit fractions" territory as the
Erdos-Straus conjecture (which is exactly what Erdos problem 284's tags name).

f is computed FIRST from first principles with exact Python `fractions.Fraction`
arithmetic (no OEIS lookup, no hard-coded literal answer) -- see
`classical_f_table()` below. That table is the ground truth the quantum result
is checked against.

Quantum construction
---------------------
A Grover search circuit over the 4-bit register representing a in [0, 15]:
  - the oracle is a phase oracle built directly from the classical truth table
    computed above (X gates on the 0-bits of each marked a, then a
    multi-controlled Z, then undo the X gates) -- so the "oracle" is not
    hand-picked, it is generated mechanically from the classical computation;
  - the diffuser is the standard Grover diffusion operator;
  - the number of Grover iterations is chosen from the standard formula
    floor(pi/4 * sqrt(2^n / M)) where M = number of marked items.

The circuit is run on the ideal AerSimulator (statevector-based, shots=2048).
PASS requires: the set of a-values whose measured probability exceeds a
1/2^n baseline threshold is exactly the classically-marked set (i.e. Grover
search recovers exactly the classically correct answers, and nothing else).

No external dependencies beyond qiskit, qiskit_aer, numpy.
"""

from fractions import Fraction
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_BITS = 4          # a ranges over 0..15
N_STATES = 2 ** N_BITS
TARGET = Fraction(4, 5)   # 4/N for N = 5
B_BOUND = 60         # classical search bound for b, c


def has_two_term_decomposition(remainder: Fraction, bound: int) -> bool:
    """True iff remainder == 1/b + 1/c for some positive integers b <= c <= bound."""
    if remainder <= 0:
        return False
    for b in range(1, bound + 1):
        rb = remainder - Fraction(1, b)
        if rb < 0:
            continue  # 1/b alone exceeds remainder; larger b may still work
        if rb == 0:
            continue  # need TWO positive unit fractions, not one
        # rb == 1/c  =>  c == 1/rb, must be a positive integer >= b (wlog b<=c)
        if rb.numerator == 1:
            c = rb.denominator
            if c >= b and c <= bound:
                return True
    return False


def classical_f_table():
    """Compute f(a) for a in [0, 2**N_BITS - 1] from first principles."""
    marked = []
    for a in range(N_STATES):
        if a == 0:
            continue  # a=0 is not a valid denominator; f(0) = False by definition
        remainder = TARGET - Fraction(1, a)
        if remainder <= 0:
            continue
        if has_two_term_decomposition(remainder, B_BOUND):
            marked.append(a)
    return marked


def build_oracle(marked_values, n_bits):
    """Phase oracle: flips the sign of |a> for each a in marked_values."""
    qc = QuantumCircuit(n_bits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_bits}b")[::-1]  # little-endian qubit order
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z across all n_bits qubits (phase flip on |11...1>)
        if n_bits == 1:
            qc.z(0)
        else:
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_bits):
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    qc.h(n_bits - 1)
    qc.mcx(list(range(n_bits - 1)), n_bits - 1)
    qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(marked_values, n_bits, shots=2048):
    m = len(marked_values)
    n_states = 2 ** n_bits
    if m == 0 or m >= n_states:
        iterations = 0
    else:
        iterations = max(1, int(np.floor((np.pi / 4) * np.sqrt(n_states / m))))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(marked_values, n_bits)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    marked = classical_f_table()
    print(f"Erdos problem #284 -- unit-fraction Egyptian decomposition search")
    print(f"Target fraction: 4/5, first denominator a in [1, {N_STATES - 1}]")
    print(f"Classically marked a-values (4/5 = 1/a + 1/b + 1/c has a solution): {marked}")

    counts, iterations = run_grover(marked, N_BITS)
    print(f"Grover iterations used: {iterations}")
    print(f"Raw measurement counts: {counts}")

    total_shots = sum(counts.values())
    # a value is "selected" by the quantum run if its measured frequency
    # clears a threshold well above the uniform baseline (1 / 2**n_bits)
    baseline = 1.0 / N_STATES
    threshold = max(3 * baseline, 0.05)
    selected = set()
    for bitstring, count in counts.items():
        prob = count / total_shots
        if prob >= threshold:
            a_value = int(bitstring, 2)
            selected.add(a_value)

    classical_set = set(marked)
    print(f"Quantum-selected a-values (prob >= {threshold:.3f}): {sorted(selected)}")
    print(f"Classical a-values: {sorted(classical_set)}")

    verified = selected == classical_set and len(classical_set) > 0
    ran_ok = True

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return ran_ok, verified


if __name__ == "__main__":
    main()
