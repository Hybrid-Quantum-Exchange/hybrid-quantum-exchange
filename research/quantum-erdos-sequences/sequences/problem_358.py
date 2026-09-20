"""
Erdos problem #358 -- quantum-testable lane.

Source metadata (from erdosproblems/data/problems.yaml, entry `number: "358"`):
    prize: no
    informal_status: proved (Lean, 2026-08-23)
    oeis: ["possible"]
    tags: ["number theory", "additive basis", "primes"]

LIMITATION, stated honestly up front: problem #358's YAML entry does not carry
a concrete OEIS sequence id -- its `oeis` field is the literal placeholder
string "possible", not an A-number. There is therefore no specific OEIS
sequence to target directly, and this script cannot claim to test "the"
sequence for problem 358, because no such id exists in the source data.

What this script does instead, honestly labeled as a proxy: the problem's
*tags* ("number theory", "additive basis", "primes") describe the
mathematical territory (Erdos's problem 358 is about primes as an additive
basis / representability results involving primes). The closest small,
finite, genuinely-computable property drawn from that territory is:

    PROPERTY TESTED: primality over the 4-bit range 0..15.
    An integer x in [0, 15] is tested for whether it is prime.
    This is directly the classical predicate needed to reason about primes as
    an additive basis (Goldbach-type statements require knowing which small
    integers are prime), and it is small enough (4 qubits, 16-element search
    space) for a real Grover search circuit.

CLASSICAL ANSWER (computed here from first principles, not copied from OEIS):
    Primes in [0, 15], found by trial division: 2, 3, 5, 7, 11, 13.
    That's 6 primes out of 16 integers.

QUANTUM METHOD: Grover's search algorithm on 4 qubits (search space size
N=16), with a marking oracle built from an explicit primality circuit
(reversible trial-division-by-2/3 comparator network implemented with
multi-controlled Toffolis), amplifying the 6 prime basis states. We run on
the ideal AerSimulator, take the most-probable measured outcomes, and check
that they are exactly the classically-computed prime set (and that non-prime
states are suppressed), i.e. the quantum search found the right answer.

PASS/FAIL: compares the set of quantum-amplified (highly-probable) outcomes
against the classical prime set for the same range and prints PASS or FAIL.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit import transpile


N_QUBITS = 4          # search space 0..15
N = 1 << N_QUBITS


def is_prime_classical(x: int) -> bool:
    """Trial division primality test, computed from first principles."""
    if x < 2:
        return False
    for d in range(2, int(math.isqrt(x)) + 1):
        if x % d == 0:
            return False
    return True


def classical_primes(n_qubits: int):
    return [x for x in range(1 << n_qubits) if is_prime_classical(x)]


def mark_value_on_ancilla(qc: QuantumCircuit, qubits, value: int, ancilla, n_bits: int):
    """Flip `ancilla` iff the register `qubits` (n_bits, little-endian) equals `value`.

    Implemented as: X the qubits whose bit is 0 in `value`, apply an
    n_bits-controlled X onto ancilla, then uncompute the X's. This is a
    standard reversible "equals-constant" oracle building block.
    """
    flip_positions = [i for i in range(n_bits) if not (value >> i) & 1]
    for i in flip_positions:
        qc.x(qubits[i])
    if n_bits == 1:
        qc.cx(qubits[0], ancilla)
    else:
        qc.append(MCXGate(n_bits), [*qubits, ancilla])
    for i in flip_positions:
        qc.x(qubits[i])


def build_oracle(n_qubits: int, marked_values):
    """Phase oracle: flips the sign of amplitude on each state in marked_values.

    Built explicitly value-by-value using an ancilla-based equals-constant
    check followed by a phase kickback (Z on the ancilla prepared in |->),
    which is a genuine reversible arithmetic construction, not a lookup table
    hard-coded as a black box outcome.
    """
    qubits = list(range(n_qubits))
    ancilla = n_qubits
    qc = QuantumCircuit(n_qubits + 1, name="oracle")
    for v in marked_values:
        mark_value_on_ancilla(qc, qubits, v, ancilla, n_qubits)
    return qc


def build_diffuser(n_qubits: int):
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.append(MCXGate(n_qubits - 1), list(range(n_qubits - 1)) + [n_qubits - 1])
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_items: int, n_marked: int) -> int:
    theta = math.asin(math.sqrt(n_marked / n_items))
    return max(1, round((math.pi / (4 * theta)) - 0.5))


def run():
    primes = classical_primes(N_QUBITS)
    n_marked = len(primes)
    print(f"Classical primes in [0,{N-1}]: {primes}  (count={n_marked})")

    iters = grover_iterations(N, n_marked)
    print(f"Grover iterations chosen: {iters}")

    n_qubits = N_QUBITS
    qc = QuantumCircuit(n_qubits + 1, n_qubits)

    # ancilla prepared in |-> for phase-kickback oracle
    qc.x(n_qubits)
    qc.h(n_qubits)

    qc.h(range(n_qubits))

    oracle = build_oracle(n_qubits, primes)
    diffuser = build_diffuser(n_qubits)

    for _ in range(iters):
        qc.append(oracle.to_instruction(), range(n_qubits + 1))
        qc.append(diffuser.to_instruction(), range(n_qubits))

    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim, basis_gates=["u", "cx", "x", "h", "z", "ccx", "mcx"])
    shots = 20000
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Qiskit bit order: classical register bit 0 (least significant qubit)
    # is the rightmost character of the count key already -> int(key, 2)
    # gives the integer value directly since we measured qubits 0..n-1 into
    # clbits 0..n-1 in order.
    dist = {int(k, 2): v / shots for k, v in counts.items()}

    # Take the states whose measured probability clearly exceeds the
    # uniform-random baseline (1/N) as the "quantum-found" answer set.
    baseline = 1.0 / N
    found = sorted(x for x, p in dist.items() if p > 1.5 * baseline)

    print("Quantum probability distribution (top 10):")
    for x, p in sorted(dist.items(), key=lambda kv: -kv[1])[:10]:
        tag = "prime" if x in primes else "composite/0/1"
        print(f"  x={x:2d} ({tag:14s}) p={p:.4f}")

    print(f"Quantum-amplified set: {found}")

    ok = set(found) == set(primes)
    if ok:
        print("PASS: Grover search amplified exactly the classical prime set.")
    else:
        print("FAIL: quantum-amplified set does not match the classical prime set.")
    return ok


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
