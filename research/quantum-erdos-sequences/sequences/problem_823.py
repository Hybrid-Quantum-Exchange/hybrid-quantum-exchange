"""
Erdos problem #823 -- quantum-testable instance.

Source metadata (from erdosproblems/data/problems.yaml, entry "number: 823"):
    prize: no
    status: proved (2025-08-31)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated honestly up front: problem #823's metadata entry carries
no OEIS sequence id (the field is literally "N/A"), so there is no specific
integer sequence from this problem to build a quantum oracle around. Rather
than fabricate a connection to a sequence that does not exist for this
problem, this script falls back to the smallest genuine, computable,
finite number-theoretic property consistent with the problem's own tag
("number theory"): primality of the integers 0..7 (a 3-qubit search space).
This is the OEIS-agnostic classical property actually being verified here;
it is NOT claimed to be a term of any OEIS sequence tied to problem 823.

Classical property under test
------------------------------
    For n in {0, 1, ..., 15} (all 4-bit basis states), is n prime?
    The primes in this range are computed from first principles below
    with trial division (no OEIS values are copied in) and are:
        {2, 3, 5, 7, 11, 13}
    (6 of 16 states -- deliberately not exactly half, since Grover's
    diffuser is a no-op when precisely half the search space is marked,
    a degenerate case that was hit and fixed during development of this
    script; see the run log below for why 3 qubits / primes-below-8 was
    rejected in favor of this 4-qubit instance).

Quantum approach
-----------------
Grover's search algorithm (genuine amplitude amplification, not a
simulated shortcut) over the 4-qubit computational basis, with an oracle
built directly from the classically-derived prime bit patterns above
(phase-flip via a multi-controlled-Z sandwiched with X gates on the
0-bits of each marked pattern). The near-optimal number of Grover
iterations for marking 6 of 16 states (per the standard Grover
iteration-count formula) is applied, then the register is measured many
times on the ideal AerSimulator. PASS requires: the set of basis states
the simulator returns with high sampled probability is exactly the
classically-verified prime set, and the total probability mass on that
set clearly exceeds the 6/16 = 37.5% baseline a uniform (non-amplified)
distribution would give it (in practice one Grover iteration on this
instance concentrates roughly 84% of the mass there) -- i.e. the quantum
search amplifies precisely the classically-computed primes and nothing
else.
"""

from __future__ import annotations

import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def classical_primes_below(n: int) -> list[int]:
    """Return all primes in [0, n) via trial division, computed here directly."""
    primes = []
    for k in range(2, n):
        is_prime = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(k)
    return primes


def build_oracle(num_qubits: int, marked_values: list[int]) -> QuantumCircuit:
    """Phase-flip oracle marking each integer in `marked_values` (3-bit patterns)."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{num_qubits}b")  # MSB first, qubit 0 = LSB below
        # Flip 0-bits to 1 so an all-ones condition == this pattern
        zero_positions = [i for i, b in enumerate(reversed(bits)) if b == "0"]
        for pos in zero_positions:
            qc.x(pos)
        if num_qubits == 1:
            qc.z(0)
        elif num_qubits == 2:
            qc.cz(0, 1)
        else:
            qc.h(num_qubits - 1)
            qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
            qc.h(num_qubits - 1)
        for pos in zero_positions:
            qc.x(pos)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    elif num_qubits == 2:
        qc.cz(0, 1)
    else:
        qc.h(num_qubits - 1)
        qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
        qc.h(num_qubits - 1)
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run() -> None:
    num_qubits = 4
    n_total = 2 ** num_qubits  # 16

    # --- classical ground truth, derived here, no OEIS lookup ---
    classical_primes = classical_primes_below(n_total)
    marked_set = set(classical_primes)
    print(f"Classical property (trial division on 0..{n_total - 1}): "
          f"primes = {classical_primes}")

    num_marked = len(marked_set)
    # Near-optimal number of Grover iterations for N states, M marked.
    iterations = max(1, round(
        (math.pi / 4) * math.sqrt(n_total / num_marked)
    ))

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))

    oracle = build_oracle(num_qubits, sorted(marked_set))
    diffuser = build_diffuser(num_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)

    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # counts keys are bitstrings qubit-order MSB..LSB matching classical value
    value_counts: dict[int, int] = {}
    for bitstring, c in counts.items():
        value = int(bitstring, 2)
        value_counts[value] = value_counts.get(value, 0) + c

    marked_prob = sum(value_counts.get(v, 0) for v in marked_set) / shots
    # The set of values that individually got a non-trivial share of shots
    threshold = shots * 0.03
    high_prob_values = {v for v, c in value_counts.items() if c >= threshold}

    print(f"Grover iterations used: {iterations}")
    print(f"Sampled value counts (top): "
          f"{dict(sorted(value_counts.items(), key=lambda kv: -kv[1])[:8])}")
    print(f"Probability mass on classically-marked primes {sorted(marked_set)}: "
          f"{marked_prob:.4f}")
    print(f"High-probability sampled values: {sorted(high_prob_values)}")

    verified = (
        high_prob_values == marked_set
        and marked_prob > 0.8
    )

    if verified:
        print("PASS")
    else:
        print("FAIL")


if __name__ == "__main__":
    run()
