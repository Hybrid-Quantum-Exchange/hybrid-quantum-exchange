"""
Erdos problem #457 -- quantum-testable instance.

OEIS sequence used: A391668.
  A391668 is a table read by antidiagonals: T(n,k) = the least positive
  integer m > 1 that is coprime to every integer in the window
  [n+1, n+k] (i.e. gcd(m, n+j) == 1 for all j = 1..k).

Classical property tested here (computed from first principles below,
not copied from OEIS):
  T(1, 2) = the least integer m > 1 with gcd(m, 2) == 1 and gcd(m, 3) == 1,
  i.e. the least integer m > 1 coprime to 6.
  By trial: 2 fails (even), 3 fails (div by 3), 4 fails (even), 5 works
  (odd, not div by 3) => T(1,2) = 5. This matches the row-1 entries of
  A391668 (3, 5, 5, 7, 7, 11, ...): the k=2 term is 5.

Quantum approach:
  We restrict the search space to the 16 integers m = 2..17, encoded as a
  4-qubit register (i = m-2, i in 0..15). We classically compute the set
  of "marked" indices i for which m = i+2 satisfies gcd(m,2)==1 and
  gcd(m,3)==1 (i.e. m coprime to 6) -- this is the same arithmetic
  predicate that defines T(1,2). We then build a genuine Grover search
  circuit whose oracle phase-flips exactly those marked basis states
  (via per-state multi-controlled-Z, X-gate sandwiched to match each
  marked bit pattern) and whose diffuser is the standard Grover diffusion
  operator, run on the ideal AerSimulator. Grover amplifies the marked
  subspace {m : m coprime to 6, 2<=m<=17} = {5,7,11,13,17}; we then take
  the minimum of the values observed with non-negligible probability and
  compare it against the classically-computed answer T(1,2) = 5.

This is a real (if small) instance of exactly the kind of coprimality
search that defines the A391668 sequence -- it is not the general open
Erdos problem, just a finite, computable, quantum-searchable instance of
the same predicate.
"""

import math
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

N_QUBITS = 4
N = 2 ** N_QUBITS  # search space size: 16 values, m = i + 2 for i in 0..15
OFFSET = 2


def is_marked_value(m: int) -> bool:
    """m is coprime to both 2 and 3 (i.e. coprime to 6)."""
    return math.gcd(m, 2) == 1 and math.gcd(m, 3) == 1


def classical_answer():
    """Classically compute T(1,2): least m>1 coprime to 2 and to 3."""
    for m in range(2, OFFSET + N):
        if is_marked_value(m):
            return m
    raise RuntimeError("no answer found in search window")


def marked_indices():
    return [i for i in range(N) if is_marked_value(i + OFFSET)]


def append_multi_controlled_z(qc: QuantumCircuit, qubits):
    """Phase-flip the |11...1> state on the given qubits (multi-controlled Z)."""
    if len(qubits) == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def append_state_oracle(qc: QuantumCircuit, qubits, index: int, n_qubits: int):
    """Phase-flip the single computational basis state |index> (little-endian)."""
    bits = [(index >> b) & 1 for b in range(n_qubits)]
    flip = [qubits[b] for b, bit in enumerate(bits) if bit == 0]
    if flip:
        qc.x(flip)
    append_multi_controlled_z(qc, qubits)
    if flip:
        qc.x(flip)


def build_oracle(n_qubits: int, marked):
    qc = QuantumCircuit(n_qubits, name="Oracle")
    qubits = list(range(n_qubits))
    for idx in marked:
        append_state_oracle(qc, qubits, idx, n_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qubits = list(range(n_qubits))
    qc.h(qubits)
    qc.x(qubits)
    append_multi_controlled_z(qc, qubits)
    qc.x(qubits)
    qc.h(qubits)
    return qc


def build_grover_circuit(n_qubits: int, marked, iterations: int):
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, marked)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_qubits))
        qc.append(diffuser.to_gate(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run():
    classical = classical_answer()
    marked = marked_indices()
    marked_values = sorted(i + OFFSET for i in marked)

    m_count = len(marked)
    optimal_iters = max(1, round((math.pi / 4) * math.sqrt(N / m_count)))

    qc = build_grover_circuit(N_QUBITS, marked, optimal_iters)

    sim = AerSimulator()
    shots = 4096
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Convert bitstrings (little-endian per Qiskit convention: c[0] is
    # rightmost) back into index -> value, and keep outcomes seen with
    # non-negligible probability.
    threshold = shots * 0.02
    observed_values = []
    for bitstring, freq in counts.items():
        if freq < threshold:
            continue
        idx = int(bitstring, 2)
        observed_values.append(idx + OFFSET)

    quantum_min = min(observed_values) if observed_values else None
    observed_are_marked = all(v in marked_values for v in observed_values)

    print(f"Classical answer T(1,2) = least m>1 coprime to 6: {classical}")
    print(f"Marked values (m in [2,17] coprime to 6): {marked_values}")
    print(f"Grover iterations used: {optimal_iters}")
    print(f"Measurement counts: {counts}")
    print(f"Observed (high-probability) values: {sorted(observed_values)}")
    print(f"Quantum-found minimum: {quantum_min}")

    ok = (
        observed_are_marked
        and quantum_min == classical
        and set(observed_values) == set(marked_values)
    )

    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys

    success = run()
    sys.exit(0 if success else 1)
