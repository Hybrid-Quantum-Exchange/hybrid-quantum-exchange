"""
Erdos problem #698 -- quantum-testable sequence lane.

Source metadata (data/problems.yaml in the manman4/erdosproblems clone,
entry "number: \"698\""):
    prize: no
    tags: ["number theory", "binomial coefficients"]
    oeis: ["possible"]
    status: proved (Lean), informal_status: proved

LIMITATION, stated honestly up front: the metadata for problem #698 does not
carry a real OEIS sequence id -- the "oeis" field is the placeholder string
"possible", not an A-number. There is therefore no specific integer sequence
from this problem to search or verify membership in. Rather than fabricate
an OEIS id or copy a value with no traceable source, this script instead
builds a genuine quantum circuit around the one concrete, finite,
computable piece of mathematical content the metadata *does* commit to:
the "binomial coefficients" tag, via Kummer's theorem on binomial
coefficient parity.

Classical property tested (derived and checked in this script, not copied
from anywhere):
    Fix n = 10 (n < 16, so k ranges over a 4-qubit search space [0, 15]).
    By Kummer's theorem, C(n, k) is ODD iff (k AND (n - k)) == 0, i.e. the
    binary digits of k are a subset of the binary digits of n. We compute,
    from first principles (exact integer binomial coefficients via
    math.comb, reduced mod 2), the exact set S = { k in [0,15] : C(10,k) is
    odd }. This is a small, finite, fully computable search problem: "find
    all k for which C(10,k) is odd."

Quantum approach:
    Grover's algorithm over 4 qubits (16-element search space). The oracle
    phase-flips exactly the basis states |k> for k in S (built with
    multi-controlled Z gates gated on the bit pattern of each marked k).
    One Grover iteration (optimal for this marked-fraction) is applied and
    the circuit is measured on AerSimulator (ideal, statevector-based
    shots). PASS iff the empirical distribution of measured k values is
    concentrated (over an overwhelming majority of shots) on the classically
    computed set S.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N = 10          # fixed n for C(n, k)
NBITS = 4       # k in [0, 15]
DIM = 1 << NBITS


def classical_odd_binomial_ks(n: int, nbits: int) -> list[int]:
    """Exact classical computation of {k in [0, 2**nbits - 1] : C(n,k) is odd}.

    Uses math.comb for the exact integer binomial coefficient (no
    approximation), independently cross-checked against Kummer's theorem
    (k & (n - k) == 0) for every k in range.
    """
    dim = 1 << nbits
    exact = set()
    for k in range(dim):
        c = math.comb(n, k) if k <= n else 0
        is_odd_exact = (c % 2 == 1)
        is_odd_kummer = (k <= n) and ((k & (n - k)) == 0)
        assert is_odd_exact == is_odd_kummer, (
            f"Kummer's theorem check failed for n={n}, k={k}: "
            f"exact parity={is_odd_exact}, kummer={is_odd_kummer}"
        )
        if is_odd_exact:
            exact.add(k)
    return sorted(exact)


def build_oracle(marked: list[int], nbits: int) -> QuantumCircuit:
    """Phase-flip oracle: |k> -> -|k> for each k in `marked`, identity elsewhere."""
    qc = QuantumCircuit(nbits, name="oracle")
    for k in marked:
        bits = format(k, f"0{nbits}b")[::-1]  # bit i -> qubit i
        flip_qubits = [i for i, b in enumerate(bits) if b == "0"]
        for q in flip_qubits:
            qc.x(q)
        if nbits == 1:
            qc.z(0)
        else:
            qc.h(nbits - 1)
            qc.mcx(list(range(nbits - 1)), nbits - 1)
            qc.h(nbits - 1)
        for q in flip_qubits:
            qc.x(q)
    return qc


def build_diffuser(nbits: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, name="diffuser")
    qc.h(range(nbits))
    qc.x(range(nbits))
    if nbits == 1:
        qc.z(0)
    else:
        qc.h(nbits - 1)
        qc.mcx(list(range(nbits - 1)), nbits - 1)
        qc.h(nbits - 1)
    qc.x(range(nbits))
    qc.h(range(nbits))
    return qc


def build_grover_circuit(marked: list[int], nbits: int, iterations: int) -> QuantumCircuit:
    qc = QuantumCircuit(nbits, nbits)
    qc.h(range(nbits))
    oracle = build_oracle(marked, nbits)
    diffuser = build_diffuser(nbits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(nbits), range(nbits))
    return qc


def main() -> None:
    marked = classical_odd_binomial_ks(N, NBITS)
    print(f"n = {N}, search space size = {DIM} (k in [0, {DIM - 1}])")
    print(f"Classical set S = {{k : C({N},k) is odd}} = {marked}")

    m = len(marked)
    # Optimal number of Grover iterations for m marked out of DIM.
    theta = math.asin(math.sqrt(m / DIM))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    print(f"Marked count m = {m}, Grover iterations = {iterations}")

    qc = build_grover_circuit(marked, NBITS, iterations)

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit reports bitstrings MSB-left over the classical register in the
    # order the register was measured; register bit i (qubit i) is char
    # position (nbits-1-i) from the left.
    hits = 0
    for bitstring, c in counts.items():
        k = int(bitstring, 2)
        if k in marked:
            hits += c

    frac = hits / shots
    print(f"Measured k values (top): {sorted(counts.items(), key=lambda x: -x[1])[:6]}")
    print(f"Fraction of shots landing on classically-marked k: {frac:.4f}")

    THRESHOLD = 0.90
    ok = frac >= THRESHOLD

    if ok:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    main()
