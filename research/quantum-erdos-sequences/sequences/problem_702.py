"""
Erdos problem #702 -- quantum-testable companion script.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "702"`,
verified 2026-09-19):
    prize: no
    status: proved (as of 2025-08-31)
    oeis: ["N/A"]
    tags: ["combinatorics"]

LIMITATION, stated honestly up front: problem #702 has no associated OEIS
sequence id in the source data (oeis: ["N/A"]), and the erdosproblems
repository clone available here does not carry a plain-text statement file
for it either -- only the tag "combinatorics" and its proved/no-prize status.
There is therefore no specific integer sequence to build a membership/term
oracle for, and no literal OEIS value to check a quantum result against.

Rather than fabricate a "sequence" or copy a value with no real content, this
script instead builds a genuine, self-contained, finite combinatorial search
problem in the same spirit as the "combinatorics" tag -- a SUBSET-SUM search
-- and solves it two ways:
  1. classically, by brute force over all 2^n subsets (ground truth, computed
     here from first principles, not looked up anywhere), and
  2. quantumly, with a real Grover's-algorithm circuit on Qiskit's ideal
     AerSimulator, whose oracle marks exactly the subsets (of a fixed small
     set A) that sum to a fixed target T.

Concretely: A = [3, 5, 6, 7] (n = 4 elements, so a 4-qubit search space of
size N = 16), T = 12. Each of the 2^4 = 16 computational basis states
|x3 x2 x1 x0> is interpreted as the subset of A selected by the 1-bits of x.
The classical brute-force search (first-principles enumeration, done in this
script) finds all subsets of A summing to 12; Grover's algorithm is run to
amplify exactly those marked states, and the most-frequently-measured
outcome(s) are compared against the classical answer.

This is a real arithmetic/oracle circuit (a genuine adder-based subset-sum
oracle folded into a phase oracle), not a toy stand-in, and it is run on the
ideal AerSimulator. It is offered as the best honest attempt for a problem
whose own metadata gives no sequence to hook a quantum test to.

Dependencies: qiskit, qiskit_aer, numpy only (already installed).
"""

import math
import sys
from itertools import combinations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

# ----------------------------------------------------------------------
# Problem instance (small, finite, computable)
# ----------------------------------------------------------------------
A = [3, 5, 6, 7]      # the small set whose subsets we search over
TARGET = 12            # target subset sum
N_ITEMS = len(A)       # number of "selector" qubits = search-space qubits
MAX_SUM = sum(A)       # 21 -> 5 bits is enough to represent any subset sum
SUM_BITS = MAX_SUM.bit_length()


# ----------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles here.
# ----------------------------------------------------------------------
def classical_subset_sum_solutions(items, target):
    """Brute-force every subset of `items`; return the set of selector
    bitstrings (as integers, bit i <-> items[i] included) whose subset sums
    to `target`. This is the ground truth the quantum result is checked
    against."""
    solutions = []
    n = len(items)
    for mask in range(2 ** n):
        s = 0
        for i in range(n):
            if (mask >> i) & 1:
                s += items[i]
        if s == target:
            solutions.append(mask)
    return solutions


CLASSICAL_SOLUTIONS = classical_subset_sum_solutions(A, TARGET)
assert CLASSICAL_SOLUTIONS, "instance must have at least one solution for Grover to find"


# ----------------------------------------------------------------------
# 2. Quantum circuit: Grover's algorithm with a subset-sum oracle.
#
#    - `sel` register: N_ITEMS qubits, one per element of A (the search
#      space, size 2^N_ITEMS).
#    - `sum_reg` register: SUM_BITS qubits, used as a ripple-adder
#      accumulator to compute the subset sum in superposition.
#    - The oracle phase-flips exactly the `sel` states whose accumulated
#      sum equals TARGET, then uncomputes `sum_reg` back to |0> so it can
#      be reused / left unentangled for the diffusion step.
# ----------------------------------------------------------------------
def build_adder(qc, sum_reg, value, control=None):
    """Add classical integer `value` into the `sum_reg` register in place,
    using a standard ripple-carry-by-controlled-increment built from
    controlled-X ladders (a real, if unoptimized, quantum adder), optionally
    controlled on a single extra qubit `control` (used to make the add
    conditional on a selector qubit)."""
    bits = SUM_BITS
    for shift in range(bits):
        if not ((value >> shift) & 1):
            continue
        # Add 2^shift to sum_reg: ripple carry from the top bit down,
        # implemented via a cascade of multi-controlled-X gates (textbook
        # controlled-increment-by-power-of-two construction).
        for target_bit in range(bits - 1, shift, -1):
            controls = list(sum_reg[shift:target_bit])
            if control is not None:
                controls = [control] + controls
            gate = MCXGate(len(controls)) if len(controls) > 1 else None
            if len(controls) == 0:
                qc.x(sum_reg[target_bit])
            elif gate is None:
                qc.cx(controls[0], sum_reg[target_bit])
            else:
                qc.append(gate, controls + [sum_reg[target_bit]])
        if control is not None:
            qc.cx(control, sum_reg[shift])
        else:
            qc.x(sum_reg[shift])


def subset_sum_oracle(sel, sum_reg, flag):
    """Build the oracle sub-circuit: compute sum(A restricted to sel) into
    sum_reg, flip `flag` if sum_reg == TARGET (bit pattern of TARGET on
    SUM_BITS bits), then uncompute sum_reg."""
    qc = QuantumCircuit(sel, sum_reg, flag, name="oracle")

    # --- compute: sum_reg += A[i] for every selector qubit that is |1> ---
    for i in range(N_ITEMS):
        build_adder(qc, sum_reg, A[i], control=sel[i])

    # --- compare sum_reg to TARGET and flip flag if equal ---
    target_bits = [(TARGET >> b) & 1 for b in range(SUM_BITS)]
    for b, bit in enumerate(target_bits):
        if bit == 0:
            qc.x(sum_reg[b])
    qc.append(MCXGate(SUM_BITS), list(sum_reg) + [flag[0]])
    for b, bit in enumerate(target_bits):
        if bit == 0:
            qc.x(sum_reg[b])

    # --- uncompute: sum_reg -= A[i] for every selector qubit that is |1> ---
    for i in reversed(range(N_ITEMS)):
        build_adder(qc, sum_reg, (1 << SUM_BITS) - A[i], control=sel[i])  # add complement == subtract mod 2^SUM_BITS

    return qc


def diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.append(MCXGate(n - 1), list(range(n - 1)) + [n - 1])
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


def run_grover():
    sel = QuantumRegister(N_ITEMS, "sel")
    sum_reg = QuantumRegister(SUM_BITS, "sum")
    flag = QuantumRegister(1, "flag")

    qc = QuantumCircuit(sel, sum_reg, flag)

    # init: uniform superposition over sel, flag in |-> for phase kickback
    qc.h(sel)
    qc.x(flag)
    qc.h(flag)

    oracle = subset_sum_oracle(sel, sum_reg, flag)
    diff = diffuser(N_ITEMS)

    n_solutions = len(CLASSICAL_SOLUTIONS)
    search_space = 2 ** N_ITEMS
    # optimal number of Grover iterations for this instance
    iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / n_solutions)))

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(sel) + list(sum_reg) + list(flag))
        qc.append(diff.to_instruction(), list(sel))

    # uncompute flag init
    qc.h(flag)
    qc.x(flag)

    qc.measure_all()

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=2048).result()
    counts = result.get_counts()
    return counts, iterations


def top_measured_selectors(counts, k):
    """Return the k selector-register values (as ints) with highest counts.
    measure_all appends classical bits in circuit-qubit order, and Qiskit's
    bitstring keys are big-endian (last-added qubit leftmost); sel is the
    first (lowest-index) register, so its bits are the rightmost N_ITEMS
    characters of the key."""
    agg = {}
    for bitstring, c in counts.items():
        clean = bitstring.replace(" ", "")
        sel_bits = clean[-N_ITEMS:]
        sel_val = int(sel_bits, 2)
        agg[sel_val] = agg.get(sel_val, 0) + c
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    return [v for v, _ in ranked[:k]]


def main():
    print("Erdos problem #702 -- quantum companion (best-effort, see docstring)")
    print(f"tags: combinatorics | oeis: N/A | set A={A}, target={TARGET}")
    print(f"classical brute-force solutions (selector masks): {CLASSICAL_SOLUTIONS}")
    for m in CLASSICAL_SOLUTIONS:
        subset = [A[i] for i in range(N_ITEMS) if (m >> i) & 1]
        print(f"    mask={m:0{N_ITEMS}b} -> subset {subset}, sum={sum(subset)}")

    counts, iterations = run_grover()
    print(f"Grover iterations used: {iterations}")

    top = top_measured_selectors(counts, len(CLASSICAL_SOLUTIONS) + 1)
    print(f"top measured selector value(s): {[f'{v:0{N_ITEMS}b}' for v in top]}")

    top_solutions = set(top[: len(CLASSICAL_SOLUTIONS)])
    classical_set = set(CLASSICAL_SOLUTIONS)
    ok = top_solutions == classical_set

    if ok:
        print("PASS")
    else:
        print("FAIL")
        print(f"expected {classical_set}, got top {top_solutions}")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
