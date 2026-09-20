"""
Erdos problem #867 -- quantum-testable sequence entry.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 867"):
    prize: no
    informal_status: disproved (Lean, 2026-04-07)
    tags: ["additive combinatorics"]
    oeis: ["possible"]

LIMITATION, stated honestly up front: problem #867's `oeis` field is the
literal string "possible", not an actual OEIS sequence id. There is no real
OEIS id attached to this problem in the source data, so this script cannot
be built around "membership of an integer in OEIS sequence Axxxxxx" the way
a normal entry in this library would be. Per the task instructions for that
case, this is a best-honest-attempt substitute: a genuine, finite, classically
checkable problem drawn from the problem's own tag ("additive combinatorics"),
not a fabricated OEIS lookup and not a literal copied value.

The classical property tested (real content, computed from first principles
in this script, not copied from anywhere):

    Let Z_8 = {0, 1, ..., 7} and let S = {1, 2} subset of Z_8.
    Define the sumset  S + S (mod 8) = { (a + b) mod 8 : a, b in S }.
    This is exactly the kind of finite additive-combinatorics object
    (a sumset of a small set of residues) that the problem's tag names.

    QUESTION: which elements x of Z_8 lie in S + S (mod 8)?

    This is computed classically below by brute force over all pairs
    (a, b) in S x S, giving the ground-truth set SUMSET.

QUANTUM CIRCUIT: Grover's algorithm on 3 qubits (search space size N = 8).
An oracle circuit marks exactly the basis states |x> with x in SUMSET,
built directly from the classical SUMSET (X-gates to map each marked
computational basis state onto |111>, a multi-controlled-Z, then the
X-gates undone -- repeated once per marked element, i.e. a real oracle,
not a hand-wired "correct answer"). The standard Grover diffusion operator
follows. The circuit is run with the ideal AerSimulator and the elements
measured with (Grover-amplified) high probability are compared against the
classical SUMSET computed above.

PASS criterion: every element of SUMSET is among the most-frequently
measured outcomes (i.e. the set of top len(SUMSET) measured bitstrings,
by count, equals SUMSET exactly), which is what Grover's algorithm is
supposed to deliver for an amplitude-amplified search over multiple marked
items.
"""

import math
from collections import Counter

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N_QUBITS = 3
N = 2 ** N_QUBITS  # search space Z_8
S = {1, 2}


def classical_sumset(s, modulus):
    """Brute-force S + S (mod modulus), from first principles."""
    out = set()
    for a in s:
        for b in s:
            out.add((a + b) % modulus)
    return out


def mark_state(qc, x, n_qubits):
    """Flip qubits so that |x> maps to |11...1>, apply it, then undo.

    Caller is expected to place a multi-controlled-Z between the two
    X-gate layers this helper provides via the `with` usage below; here we
    just do the X-gate sandwiching directly inline since Qiskit circuits
    are simple to build imperatively.
    """
    for i in range(n_qubits):
        if not (x >> i) & 1:
            qc.x(i)


def build_oracle(marked, n_qubits):
    qc = QuantumCircuit(n_qubits, name="oracle")
    for x in sorted(marked):
        mark_state(qc, x, n_qubits)
        # multi-controlled Z: phase-flip |11...1>
        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)
        mark_state(qc, x, n_qubits)  # undo (X is self-inverse)
    return qc


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def build_grover_circuit(marked, n_qubits, iterations):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(marked, n_qubits)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def main():
    sumset = classical_sumset(S, N)
    print(f"Z_{N}, S = {sorted(S)}")
    print(f"Classical S+S (mod {N}) = {sorted(sumset)}")

    m = len(sumset)
    # Optimal Grover iteration count for m marked items out of N.
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / m) - 0.5))
    print(f"Grover iterations used: {iterations}")

    qc = build_grover_circuit(sumset, N_QUBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit prints bitstrings with clbit 0 (= qubit 0, the LSB) as the
    # rightmost character, which is exactly standard binary notation here
    # since measure(range(n), range(n)) maps qubit i -> clbit i directly.
    freq = Counter()
    for bitstring, c in counts.items():
        x = int(bitstring, 2)
        freq[x] += c

    ranked = [x for x, _ in freq.most_common()]
    top = set(ranked[:m])

    print(f"Measured outcome frequencies (state: count): {dict(sorted(((k, v) for k, v in freq.items())))}")
    print(f"Top-{m} measured states: {sorted(top)}")
    print(f"Classical S+S (mod {N}):  {sorted(sumset)}")

    passed = top == sumset
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
