"""
Erdos problem #541 — quantum-testable lane (best-honest-attempt, limited).

Source metadata (data/problems.yaml, manman4/erdosproblems, entry `number: "541"`):
    prize: no
    informal_status: proved (Lean, 2025-12-30)
    oeis: ["N/A"]
    tags: ["number theory"]

LIMITATION, stated up front: problem #541 has no OEIS sequence id attached
(oeis: ["N/A"]) and the cloned dataset carries no problem statement/title
text beyond the tag "number theory" — there is no numerical sequence to
derive a property from, so no property genuinely specific to problem #541
can be built or verified here. Per instructions, this script is the best
honest attempt rather than a fabricated pass: it builds a REAL, small,
genuinely computed quantum circuit for a bona fide finite number-theory
search problem (the tag problem #541 does carry), and is explicit that the
instance is *generic to the tag*, not derived from problem #541's own
(unavailable) content.

Chosen property (finite, computable, classically checked first):
    "Grover search over N = 2^n = 16 candidates (n = 4 qubits) for the
    unique x in [0, 15] such that x is prime, x mod 4 == 3, AND x > 10
    (this narrowing is needed only to make the classical answer set a
    singleton, which a single-target Grover oracle requires; the
    underlying predicate -- primality combined with a residue class --
    is still a genuine, non-fabricated number-theory property)."
    This is an ordinary decidable number-theory predicate over a small
    finite domain -- exactly the shape of object ("small search space
    whose answer is a known term") the task allows when no OEIS id exists.

Classical computation (done here, first principles, no OEIS lookup):
    We test primality by trial division for every x in 0..15 and check
    x % 4 == 3, and print/assert the resulting classical answer set before
    ever touching Qiskit.

Quantum computation:
    A 4-qubit Grover search (oracle built from a MCZ over the exact bit
    pattern of the *unique* marked element found classically, plus the
    standard diffuser), run on qiskit_aer's AerSimulator (statevector +
    shots), with the correct number of Grover iterations for this N and
    marked-count computed from the classical answer, comparing the
    highest-probability measured bitstring against the classical answer.

Result semantics: this verifies a real, non-trivial quantum computation
against an independently derived classical answer, but it is NOT a
verification of anything mathematically specific to Erdos problem #541,
because #541 supplies no sequence/oeis id to attach one to. `ran_ok` and
`verified_against_classical` are reported honestly against the generic
instance actually built.
"""

import sys
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCXGate


def is_prime(x: int) -> bool:
    if x < 2:
        return False
    for d in range(2, int(x ** 0.5) + 1):
        if x % d == 0:
            return False
    return True


def classical_search(n_bits: int):
    """Return the sorted list of x in [0, 2**n_bits) with x prime and x % 4 == 3."""
    N = 1 << n_bits
    hits = [x for x in range(N) if is_prime(x) and x % 4 == 3 and x > 10]
    return hits


def build_oracle(n_bits: int, marked: int) -> QuantumCircuit:
    """Phase-flip oracle marking the single computational basis state `marked`."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(marked, f"0{n_bits}b")[::-1]  # little-endian qubit order
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        qc.h(n_bits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))
    if n_bits == 1:
        qc.z(0)
    else:
        qc.h(n_bits - 1)
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
        qc.h(n_bits - 1)
    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def main() -> bool:
    n_bits = 4  # N = 16 candidates, x in [0, 15]

    classical_hits = classical_search(n_bits)
    print(f"Classical search over x in [0,{(1 << n_bits) - 1}]: "
          f"prime(x) and x % 4 == 3 and x > 10 -> {classical_hits}")

    if len(classical_hits) != 1:
        print(f"FAIL: expected exactly one classical hit for a single-target "
              f"Grover oracle, found {classical_hits}")
        return False

    marked = classical_hits[0]
    print(f"Unique classical target: x = {marked} "
          f"(binary {format(marked, f'0{n_bits}b')})")

    import math
    N = 1 << n_bits
    iterations = max(1, round((math.pi / 4) * math.sqrt(N / 1)))
    print(f"Grover iterations: {iterations}")

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    shots = 4096
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    # Most frequent measured bitstring -> integer (Qiskit returns bit order
    # c[n-1]...c[0], i.e. big-endian string of little-endian qubit indices).
    best_bits = max(counts, key=counts.get)
    measured = int(best_bits, 2)
    confidence = counts[best_bits] / shots

    print(f"Measurement counts (top 5): "
          f"{sorted(counts.items(), key=lambda kv: -kv[1])[:5]}")
    print(f"Most probable measured value: {measured} "
          f"(confidence {confidence:.3f} over {shots} shots)")

    verified = (measured == marked) and (confidence > 0.5)

    print(f"Classical answer: {marked}; Quantum (Grover) answer: {measured}")
    if verified:
        print("PASS")
    else:
        print("FAIL")
    return verified


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
