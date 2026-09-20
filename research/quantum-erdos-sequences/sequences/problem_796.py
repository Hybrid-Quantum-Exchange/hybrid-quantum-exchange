"""
Erdos problem #796 -- quantum-testable sequence lane.

Source record (erdosproblems.com data, /home/user/manman4/erdosproblems/data/
problems.yaml, entry "number: \"796\""):

    number: "796"
    prize: "no"
    informal_status: { state: "open", last_update: "2025-08-31" }
    formal_status:   { state: "unformalized" }
    status:          { state: "open", last_update: "2025-08-31" }
    oeis: ["possible"]
    formalized:      { state: "yes", last_update: "2026-08-07" }
    tags: ["number theory"]

LIMITATION (read before trusting the "PASS" below):
The oeis field for problem 796 is the literal string "possible" -- this is
a placeholder the erdosproblems.com dataset uses for "an OEIS sequence may
exist but has not been identified/linked", not an actual OEIS A-number.
There is no dedicated write-up file for #796 in the cloned repository
(searched data/ and *.md; only a passing mention in README.md, which is not
problem-specific), and no other OEIS id is given. So there is no genuine,
identifiable OEIS sequence tied to this problem in the available source --
picking a specific A-number for #796 here would mean fabricating a link
between the problem and a sequence, which the task instructions explicitly
forbid.

Because of that, this script cannot build a circuit that tests a property
*of Erdos problem #796's own sequence*, honestly. Per the task's fallback
instructions ("if no genuine quantum circuit can be constructed ... write
the script anyway with your best honest attempt, note the limitation
clearly"), what follows is a best-honest-attempt substitute: a real,
self-contained Qiskit circuit exercising the one concrete fact the record
does give us -- problem 796 is tagged "number theory" -- via Grover search
for prime numbers in a small range. This is a genuine, independently
verifiable quantum computation (not a fabricated OEIS value), but it is
*not* a verified test of problem 796's actual content, because that content
(and its OEIS sequence, if any) is not available from the cloned data.

Chosen finite instance (independent of the missing OEIS id):
    N = 16 (4 qubits: |0..15>).
    Property tested: "n is prime" (2, 3, 5, 7, 11, 13 are the primes < 16).
    The classical answer is computed here from first principles (trial
    division), not copied from any table.

Circuit: Grover's algorithm.
    - 4-qubit oracle flags exactly the primality bit pattern computed
      classically (implemented as a phase oracle keyed to the classical
      set of prime indices -- a real multi-controlled-Z oracle per marked
      state, not a black box).
    - Optimal number of Grover iterations computed from the classical
      count of marked items (6 primes out of 16).
    - Run on the ideal AerSimulator; the most frequently measured
      computational basis states must be exactly the classically computed
      prime set.

PASS/FAIL: the script prints PASS iff the set of basis states receiving the
top measurement probability mass (as many states as there are classical
primes) equals the classically computed prime set exactly.
"""

import math
from itertools import combinations

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import MCMTGate, ZGate


def classical_primes(n: int) -> list[int]:
    """Trial-division primality test, computed from first principles."""
    primes = []
    for k in range(2, n):
        is_p = True
        for d in range(2, int(math.isqrt(k)) + 1):
            if k % d == 0:
                is_p = False
                break
        if is_p:
            primes.append(k)
    return primes


def build_oracle(num_qubits: int, marked_states: list[int]) -> QuantumCircuit:
    """Phase oracle flipping the sign of each marked computational basis state."""
    qc = QuantumCircuit(num_qubits, name="oracle")
    for state in marked_states:
        bits = format(state, f"0{num_qubits}b")[::-1]  # qubit0 = LSB
        zero_positions = [i for i, b in enumerate(bits) if b == "0"]
        if zero_positions:
            qc.x(zero_positions)
        if num_qubits == 1:
            qc.z(0)
        else:
            qc.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
        if zero_positions:
            qc.x(zero_positions)
    return qc


def build_diffuser(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    if num_qubits == 1:
        qc.z(0)
    else:
        qc.append(MCMTGate(ZGate(), num_qubits - 1, 1), list(range(num_qubits)))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def run_grover(num_qubits: int, marked_states: list[int], shots: int = 20000):
    n_items = 2 ** num_qubits
    n_marked = len(marked_states)

    # Optimal number of Grover iterations for this marked-set size.
    theta = math.asin(math.sqrt(n_marked / n_items))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))

    oracle = build_oracle(num_qubits, marked_states)
    diffuser = build_diffuser(num_qubits)

    qc = QuantumCircuit(num_qubits, num_qubits)
    qc.h(range(num_qubits))
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diffuser, inplace=True)
    qc.measure(range(num_qubits), range(num_qubits))

    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()
    return counts, iterations


def main() -> None:
    N = 16
    num_qubits = 4

    primes = classical_primes(N)
    print(f"Classical property: primes in [0, {N}) via trial division = {primes}")

    counts, iterations = run_grover(num_qubits, primes)
    print(f"Grover iterations used: {iterations}")

    # Take the top len(primes) measured basis states by count.
    sorted_states = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    top_k = sorted_states[: len(primes)]
    measured_ints = sorted(int(bitstring, 2) for bitstring, _ in top_k)

    total_shots = sum(counts.values())
    marked_mass = sum(c for b, c in counts.items() if int(b, 2) in primes)
    print(f"Top-{len(primes)} measured states: {measured_ints}")
    print(f"Probability mass on marked (prime) states: {marked_mass / total_shots:.4f}")

    verified = measured_ints == primes

    if verified:
        print("PASS")
    else:
        print("FAIL")

    print(
        "NOTE: this circuit verifies primality via Grover search on a small "
        "instance chosen because problem #796's OEIS field is the placeholder "
        "'possible' (no real OEIS id available in the cloned data), so no "
        "genuine sequence-specific property of #796 could be derived. See "
        "module docstring for the full limitation."
    )


if __name__ == "__main__":
    main()
