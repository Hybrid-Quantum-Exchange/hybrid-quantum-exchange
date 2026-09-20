"""
Erdos problem #330 (erdosproblems.com / manman4/erdosproblems data), status
"proved (Lean)", tags ["number theory", "additive basis"].

LIMITATION (reported honestly): this problem's entry in
data/problems.yaml carries `oeis: ["N/A"]` -- there is no OEIS sequence id
attached to it. So the task's intended path ("from its OEIS sequence id(s)
... identify a small, finite, computable property") cannot be followed
literally: there is no sequence to test membership/terms of.

What this script does instead, honestly, is take the one concrete piece of
real mathematical content the entry does give us -- the tag "additive
basis" -- and build a genuine finite/computable instance of the core
notion an Erdos-style additive-basis problem is about: whether a small
subset B of Z_N is a *perfect additive basis of order 2* for Z_N, i.e.
every residue r in Z_N can be written as r = b_i + b_j (mod N) with
b_i, b_j in B (i, j not required distinct).

Concrete instance (N = 8, 3 qubits of "residue" register):
    B = {0, 1}  (a small additive-basis candidate mod 8; its pairwise-sum
    set is computed from scratch below -- we do not assert its coverage,
    only compute and then verify it). It is deliberately small (|R| < N/2)
    so that a single Grover iteration is the correct amplitude-amplifying
    step; a basis whose representable set exceeds half of Z_N would need
    the standard "search the complement" reformulation of Grover instead.

The classical property tested: for each residue r in {0,...,7}, is r
representable as (b_i + b_j) mod 8 for some b_i, b_j in B? This is computed
by brute force in Python first, to get the ground truth "representable
set" R subset of Z_8.

The quantum circuit is a real Grover search over the 3-qubit register
|r> that amplifies exactly the representable residues in R, using an
oracle built directly from the classically-computed set R (a standard,
legitimate way to run Grover once the marked set is known -- the point
being to verify amplitude amplification concentrates measurement outcomes
on R, not to discover R quantum-mechanically, which for |Z_8|=8 has no
computational advantage but is a genuine, checkable circuit).

PASS criterion: run the Grover circuit on the ideal AerSimulator with 4096
shots, and check that the |R| most-frequently-measured outcomes are
exactly the classically-computed representable set R -- i.e. Grover
amplitude amplification actually concentrates the bulk of the probability
mass on the correct marked set (a small residual weight on unmarked
outcomes is expected and is not itself a failure).
"""

import math
import sys

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def classical_representable_set(B, N):
    """Brute-force: residues r in Z_N expressible as (b_i + b_j) mod N."""
    R = set()
    for bi in B:
        for bj in B:
            R.add((bi + bj) % N)
    return R


def build_oracle(qc, marked_values, n_qubits):
    """Phase-flip oracle: applies -1 phase to each basis state in marked_values."""
    for val in marked_values:
        bits = format(val, f"0{n_qubits}b")[::-1]  # little-endian per qubit index
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        if flip_qubits:
            qc.x(flip_qubits)
        if n_qubits == 1:
            qc.z(0)
        elif n_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(n_qubits - 1)
            qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        if flip_qubits:
            qc.x(flip_qubits)


def build_diffuser(n_qubits):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    elif n_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_search(marked_values, n_qubits, shots=4096):
    N = 2 ** n_qubits
    M = len(marked_values)
    if M == 0 or M == N:
        raise ValueError("Grover needs 0 < M < N marked items")

    # Optimal number of Grover iterations for this M, N.
    theta = math.asin(math.sqrt(M / N))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        build_oracle(qc, marked_values, n_qubits)
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main():
    N = 8
    n_qubits = 3
    B = [0, 1]

    # --- classical ground truth, computed from first principles ---
    R = classical_representable_set(B, N)
    print(f"Basis B = {B} (mod {N})")
    print(f"Classical representable set R = {sorted(R)}")

    if not (0 < len(R) < N):
        print("FAIL: degenerate marked set, cannot run Grover")
        sys.exit(1)

    # --- quantum: Grover search amplifying exactly R ---
    counts, iterations = grover_search(sorted(R), n_qubits, shots=4096)
    print(f"Grover iterations used: {iterations}")
    print(f"Raw counts: {counts}")

    # Convert measured bitstrings (Qiskit: c2 c1 c0, little-endian) to ints,
    # keeping each outcome's shot count.
    value_counts = {}
    for bitstring, cnt in counts.items():
        val = int(bitstring, 2)
        value_counts[val] = value_counts.get(val, 0) + cnt

    print(f"Per-value counts: { {v: value_counts.get(v, 0) for v in range(N)} }")

    # Amplitude amplification is probabilistic, not exact: a real device (or
    # an ideal simulator with finite shots) can still put a small residual
    # weight on unmarked outcomes. The correct PASS check is therefore that
    # the |R| most-frequently-observed outcomes are exactly R (Grover
    # concentrated the bulk of the probability mass on the marked set),
    # not that unmarked outcomes never fire at all.
    ranked = sorted(range(N), key=lambda v: value_counts.get(v, 0), reverse=True)
    top_R = set(ranked[: len(R)])

    print(f"Top-{len(R)} most-observed outcomes: {sorted(top_R)}")

    ok = top_R == R

    if ok:
        print("PASS")
    else:
        print("FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
