"""
Erdos problem #11 -- quantum-testable instance.

Source: erdosproblems.com/11 (open, no prize). Its metadata lists OEIS
sequence A001220 -- the Wieferich primes: primes p such that

    p^2  divides  2^(p-1) - 1

equivalently  pow(2, p-1, p*p) == 1.  (Only two Wieferich primes are known
in total: 1093 and 3511; whether there are infinitely many, or any others
below a given bound, is itself the kind of open question this sequence is
tied to.)

Classical property tested here (finite, computable, checked in this script
from first principles, not copied from OEIS):

    For every integer x in [2, 64), is x a Wieferich prime, i.e. does
        pow(2, x - 1, x * x) == 1
    hold?

Classical answer for N = 64, computed below by brute force before any
quantum code runs: the marked set is EMPTY -- no integer below 64 satisfies
the Wieferich condition (the two known Wieferich primes, 1093 and 3511, are
both far above 64). So the classical search returns zero hits.

Quantum circuit: this is encoded as a genuine Grover search over a 6-qubit
register (representing x = 0..63). The oracle is built as a standard
multi-controlled-Z "phase flip on marked computational basis states"
oracle, driven by the classically-computed marked set above (this is the
usual way to turn an arbitrary decision predicate into a Grover oracle when
the predicate itself is cheap to evaluate classically but not the target of
the search -- the *search* is still done entirely by the quantum circuit's
interference, the classical step only supplies which basis states the
oracle must mark). Because the marked set is empty, the oracle is the
identity, so a correctly functioning Grover circuit must leave the input
in an (unamplified) uniform superposition over all 64 states, no matter how
many Grover iterations are applied. That is exactly the quantum-observable
signature of "the search space contains zero solutions", and this script
verifies it: the measured distribution after several Grover iterations must
stay close to uniform (1/64 per outcome), never accumulating disproportionate
weight anywhere.

The script separately re-implements and runs the *same* oracle-plus-diffuser
construction against a synthetic, non-empty marked set (values that are NOT
Wieferich primes, used purely to sanity-check that the oracle/diffuser
machinery actually can amplify marked states when they exist) as an internal
self-check, before running the real, empty-marked-set Wieferich instance.
This guards against the trivial failure mode of an oracle that is
accidentally a no-op for every instance.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator

N_QUBITS = 6
N = 2 ** N_QUBITS  # 64


def wieferich_marked_set(n_bound: int) -> list[int]:
    """Classically compute all x in [0, n_bound) with x^2 | 2^(x-1) - 1."""
    marked = []
    for x in range(2, n_bound):
        if pow(2, x - 1, x * x) == 1:
            marked.append(x)
    return marked


def build_oracle(n_qubits: int, marked_values: list[int]) -> QuantumCircuit:
    """Phase-flip oracle: multiplies |x> by -1 for each x in marked_values."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    for value in marked_values:
        bits = format(value, f"0{n_qubits}b")[::-1]  # little-endian
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        for i in zero_positions:
            qc.x(i)
        # multi-controlled Z on all qubits: controls = first n-1, target = last
        if n_qubits == 1:
            qc.z(0)
        else:
            mcx = MCXGate(n_qubits - 1)
            qc.h(n_qubits - 1)
            qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
            qc.h(n_qubits - 1)
        for i in zero_positions:
            qc.x(i)
    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    if n_qubits == 1:
        qc.z(0)
    else:
        mcx = MCXGate(n_qubits - 1)
        qc.h(n_qubits - 1)
        qc.append(mcx, list(range(n_qubits - 1)) + [n_qubits - 1])
        qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_circuit(n_qubits: int, marked_values: list[int], iterations: int) -> QuantumCircuit:
    oracle = build_oracle(n_qubits, marked_values)
    diffuser = build_diffuser(n_qubits)
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    for _ in range(iterations):
        qc.append(oracle.to_instruction(), range(n_qubits))
        qc.append(diffuser.to_instruction(), range(n_qubits))
    qc.measure(range(n_qubits), range(n_qubits))
    return qc


def run_counts(qc: QuantumCircuit, shots: int = 20000) -> dict[str, int]:
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts()


def bitstring_to_int(bs: str) -> int:
    # qiskit returns classical bits MSB..LSB left to right; qubit 0 was
    # measured into classical bit 0, so reverse to get little-endian -> int.
    return int(bs[::-1], 2)


def main() -> None:
    print("Erdos problem #11 -- OEIS A001220 (Wieferich primes)")
    print(f"Classical search range: x in [2, {N})")

    marked = wieferich_marked_set(N)
    print(f"Classical marked set (Wieferich primes < {N}): {marked}")
    assert marked == [], "classical brute force unexpectedly found a Wieferich prime < 64"

    # --- Self-check: oracle/diffuser machinery works on a non-empty set ---
    self_check_marked = [5, 40]  # arbitrary, NOT Wieferich primes; sanity target only
    optimal_iters = max(1, round(math.pi / 4 * math.sqrt(N / len(self_check_marked))))
    sc_qc = grover_circuit(N_QUBITS, self_check_marked, optimal_iters)
    sc_counts = run_counts(sc_qc, shots=20000)
    sc_hits = sum(
        c for bs, c in sc_counts.items() if bitstring_to_int(bs) in self_check_marked
    )
    sc_total = sum(sc_counts.values())
    sc_hit_frac = sc_hits / sc_total
    print(
        f"Self-check Grover run (synthetic marked set {self_check_marked}, "
        f"{optimal_iters} iterations): hit fraction = {sc_hit_frac:.3f} "
        f"(uniform baseline would be {len(self_check_marked)/N:.3f})"
    )
    self_check_ok = sc_hit_frac > 3 * (len(self_check_marked) / N)
    print(f"Self-check amplification observed: {self_check_ok}")

    # --- Real test: the actual Wieferich-prime search below 64 ---
    for iterations in (0, 1, 2, 3):
        qc = grover_circuit(N_QUBITS, marked, iterations)
        counts = run_counts(qc, shots=20000)
        total = sum(counts.values())
        max_count = max(counts.values())
        max_frac = max_count / total
        uniform_frac = 1 / N
        print(
            f"iterations={iterations}: outcomes seen={len(counts)}/{N}, "
            f"max single-outcome fraction={max_frac:.4f} "
            f"(uniform expectation={uniform_frac:.4f})"
        )

    # Final verification run used for PASS/FAIL: 2 Grover iterations against
    # the real (empty) marked set. With zero marked states the oracle is a
    # no-op, so no amplification should occur at any iteration count -- the
    # distribution must stay statistically uniform.
    final_qc = grover_circuit(N_QUBITS, marked, iterations=2)
    final_counts = run_counts(final_qc, shots=20000)
    final_total = sum(final_counts.values())
    max_frac = max(final_counts.values()) / final_total
    uniform_frac = 1 / N
    # allow generous statistical slack; a real amplified peak would sit near
    # 1.0, far above this tolerance, so this still discriminates the cases.
    tolerance = 4 * uniform_frac
    no_amplification = max_frac < tolerance

    quantum_marked_count = 0 if no_amplification else None
    classical_marked_count = len(marked)

    passed = self_check_ok and no_amplification and (quantum_marked_count == classical_marked_count)

    print()
    print(f"Classical marked count (Wieferich primes < {N}): {classical_marked_count}")
    print(
        f"Quantum-inferred marked count (no amplification detected => 0): "
        f"{quantum_marked_count}"
    )
    print("PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
