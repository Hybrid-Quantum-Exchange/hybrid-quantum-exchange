"""
Erdos problem #114 (erdosproblems.com), quantum-testable lane.

Source metadata (from manman4/erdosproblems data/problems.yaml, entry
`number: "114"`): prize "$250", status "falsifiable", tags
["polynomials", "analysis"], oeis: ["N/A"]. There is NO OEIS sequence id
attached to this problem -- the listed value is the literal string "N/A".
Per the task instructions, an honest attempt is written here rather than a
fabricated sequence property, and the limitation is stated explicitly:
this script does NOT test membership in, or any term of, an actual
Erdos-problem-114 OEIS sequence, because none exists in the source data,
and it makes no claim about the actual Erdos problem #114 statement
itself (which concerns real polynomials and is not a finite/computable
search problem as posed).

Given the problem's tags ("polynomials", "analysis"), the best small,
finite, computable stand-in genuinely in that spirit is an integer-root
search for a fixed low-degree polynomial:

    Property tested: for a fixed integer polynomial
        p(x) = x^2 - 5x + 6
    and the finite domain x in {0, 1, ..., 7} (encoded as a 3-qubit
    index), find every x in that domain with p(x) == 0. This is a real,
    well-defined finite root-search problem over integer polynomials --
    squarely in the "polynomials" tag's spirit -- with a small search
    space (N = 8) suitable for Grover's algorithm.

Classical answer (computed here in the script, from first principles):
evaluate p(x) for every x in 0..7 and collect the roots. p(x) = x^2-5x+6
factors as (x-2)(x-3), so the roots in range are x = 2 and x = 3; this is
verified by direct evaluation in `classical_roots`, not asserted from
memory.

Quantum approach: Grover's search algorithm on 3 qubits (search space
size N = 8), oracle marks exactly the basis states x with p(x) == 0
(computed classically to build the oracle, since Qiskit's oracle
construction here is a phase oracle keyed on the classically-known
marked set -- the search itself, i.e. which states end up amplified, is
still checked purely from the quantum measurement statistics), diffusion
operator amplifies the marked amplitudes, and the ideal AerSimulator
counts are checked against the classical enumeration: the top-2 measured
outcomes must be exactly {2, 3} and each must outcount every unmarked
outcome.

Limitation: this is a faithful small Grover search over a polynomial
integer-root property, not a test of a specific documented OEIS integer
sequence (none is listed for problem #114), and it is not a resolution
or restatement of the actual (open, $250, falsifiable) Erdos problem
#114. ran_ok and verified_against_classical are reported honestly for
what this script actually checks.
"""

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


N_BITS = 3
N = 1 << N_BITS  # 8


def p(x: int) -> int:
    """The fixed test polynomial p(x) = x^2 - 5x + 6."""
    return x * x - 5 * x + 6


def classical_roots(n: int):
    """Enumerate all x in 0..n-1 with p(x) == 0, by direct evaluation."""
    return [x for x in range(n) if p(x) == 0]


def build_oracle_phase(marked, total_qubits: int) -> QuantumCircuit:
    """Phase oracle: applies -1 phase to each marked basis state."""
    qc = QuantumCircuit(total_qubits, name="oracle")
    for x in marked:
        bits = format(x, f"0{total_qubits}b")[::-1]  # qubit i <- bits[i]
        zero_qubits = [q for q, b in enumerate(bits) if b == "0"]
        for q in zero_qubits:
            qc.x(q)

        qc.h(total_qubits - 1)
        if total_qubits - 1 > 0:
            qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
        else:
            qc.z(0)
        qc.h(total_qubits - 1)

        for q in zero_qubits:
            qc.x(q)
    return qc


def build_diffuser(total_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(total_qubits, name="diffuser")
    qc.h(range(total_qubits))
    qc.x(range(total_qubits))
    qc.h(total_qubits - 1)
    if total_qubits - 1 > 0:
        qc.append(MCXGate(total_qubits - 1), list(range(total_qubits - 1)) + [total_qubits - 1])
    else:
        qc.z(0)
    qc.h(total_qubits - 1)
    qc.x(range(total_qubits))
    qc.h(range(total_qubits))
    return qc


def run():
    marked = classical_roots(N)
    assert marked, "classical search found no roots in range -- instance is degenerate"
    print(f"Classical answer: p(x) = x^2 - 5x + 6, roots of p(x)=0 for x in 0..{N-1}:")
    for x in marked:
        print(f"  x={x}, p(x)={p(x)}")

    num_marked = len(marked)
    theta = math.asin(math.sqrt(num_marked / N))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle_phase(marked, N_BITS)
    diffuser = build_diffuser(N_BITS)

    qc = QuantumCircuit(N_BITS, N_BITS)
    qc.h(range(N_BITS))
    for _ in range(optimal_iters):
        qc.append(oracle.to_instruction(), range(N_BITS))
        qc.append(diffuser.to_instruction(), range(N_BITS))
    qc.measure(range(N_BITS), range(N_BITS))

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=4096, seed_simulator=42).result()
    counts = result.get_counts()

    # counts keys are printed MSB..LSB over the classical bits, which here
    # map directly to qubits N_BITS-1 .. 0 (measure(range(N_BITS), range(N_BITS))).
    def bitstring_to_x(bs: str) -> int:
        bits = bs[::-1]  # index 0 == qubit0
        return int(bits[::-1], 2)

    x_counts = {}
    for bs, c in counts.items():
        x = bitstring_to_x(bs)
        x_counts[x] = x_counts.get(x, 0) + c

    sorted_x = sorted(x_counts.items(), key=lambda kv: -kv[1])
    print("\nTop measured x values by frequency:")
    for x, c in sorted_x[:8]:
        marker = " <-- MARKED (root)" if x in marked else ""
        print(f"  x={x}: {c}{marker}")

    top_n = sorted_x[:num_marked]
    top_x = set(x for x, _ in top_n)
    marked_set = set(marked)

    all_marked_on_top = top_x == marked_set
    min_marked_count = min(x_counts.get(x, 0) for x in marked_set)
    max_unmarked_count = max(
        (c for x, c in x_counts.items() if x not in marked_set), default=0
    )
    dominance_ok = min_marked_count > max_unmarked_count

    verified = all_marked_on_top and dominance_ok

    print(f"\nGrover iterations used: {optimal_iters}")
    print(f"Marked roots: {sorted(marked_set)}")
    print(f"Top-{num_marked} measured x values match root set exactly: {all_marked_on_top}")
    print(f"Every marked root outcounts every unmarked value: {dominance_ok}")

    if verified:
        print("\nPASS")
    else:
        print("\nFAIL")

    return verified


if __name__ == "__main__":
    ok = run()
    if not ok:
        raise SystemExit(1)
