"""
Erdos problem #106 -- quantum-testable-sequence lane.

Source metadata (from erdosproblems.com data, /home/user/manman4/erdosproblems/
data/problems.yaml, block "number: \"106\""):
    prize: no
    informal_status: disproved (Lean formal proof, last_update 2025-08-31)
    oeis: ["N/A"]
    tags: ["geometry"]

LIMITATION (reported honestly, per task instructions):
Problem #106 carries NO OEIS sequence id -- its `oeis` field is literally
["N/A"]. There is therefore no integer sequence attached to this problem
that a quantum circuit could search, count, or verify membership in. Every
other ingredient the task asks for (an OEIS-derived finite computable
property, a "known term" to check against) is simply absent for this
problem. Fabricating an OEIS-linked property here would violate the "do not
fabricate a property with no real mathematical content" instruction, so this
script does not attempt to encode problem #106's actual (geometric,
continuous-space) statement into a quantum oracle.

Instead, to still produce a genuine, runnable, self-contained Qiskit
artifact for this lane (rather than nothing), the script below performs a
real Grover's-algorithm search on a small, fully-specified, honestly-labeled
toy instance: finding the unique 3-bit index n in [0, 8) such that n is
divisible by 3 (i.e. n in {0, 3, 6}, target marked = 3). This is a classical
divisibility/search property of the kind the task suggests as a fallback
category ("a small search space whose answer is a known term") -- but it is
explicitly NOT a property of Erdos problem #106's mathematics, because no
such small computable property exists for this problem in the source data.
The classical answer (which indices satisfy n % 3 == 0) is computed from
first principles in this script before the circuit is built, and the
circuit's measured results are compared against that classical computation.

Honest verdict for this lane: ran_ok=True (the circuit runs and matches the
classical computation for the toy divisibility search), but
verified_against_classical is reported against that toy search, NOT against
any content of Erdos problem #106, since problem #106 has no OEIS id and no
extractable finite computable property to verify against. Do not read the
PASS below as a statement about problem #106 itself.
"""

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator, MCXGate
import numpy as np


def classical_search(n_bits: int, divisor: int):
    """Return the set of n in [0, 2**n_bits) with n % divisor == 0, computed
    directly from first principles (brute-force enumeration)."""
    universe = range(2 ** n_bits)
    return sorted(n for n in universe if n % divisor == 0)


def build_oracle(n_bits: int, marked: int) -> QuantumCircuit:
    """Phase-flip oracle that marks the single computational basis state
    |marked> (an n_bits-bit integer) with a -1 phase."""
    qc = QuantumCircuit(n_bits, name="oracle")
    bits = format(marked, f"0{n_bits}b")[::-1]  # little-endian qubit order
    zero_positions = [i for i, b in enumerate(bits) if b == "0"]

    for i in zero_positions:
        qc.x(i)

    # Multi-controlled Z: flip phase of |11...1> using an ancilla-free MCX
    # into the phase via H-MCX-H trick on the last qubit.
    qc.h(n_bits - 1)
    if n_bits - 1 == 0:
        qc.z(0)
    else:
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
    qc.h(n_bits - 1)

    for i in zero_positions:
        qc.x(i)

    return qc


def build_diffuser(n_bits: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean)."""
    qc = QuantumCircuit(n_bits, name="diffuser")
    qc.h(range(n_bits))
    qc.x(range(n_bits))

    qc.h(n_bits - 1)
    if n_bits - 1 == 0:
        qc.z(0)
    else:
        qc.append(MCXGate(n_bits - 1), list(range(n_bits - 1)) + [n_bits - 1])
    qc.h(n_bits - 1)

    qc.x(range(n_bits))
    qc.h(range(n_bits))
    return qc


def run_grover(n_bits: int, marked: int, shots: int = 2048):
    n_states = 2 ** n_bits
    # Optimal number of Grover iterations for a single marked item.
    iterations = max(1, round((np.pi / 4) * np.sqrt(n_states)))

    qc = QuantumCircuit(n_bits, n_bits)
    qc.h(range(n_bits))

    oracle = build_oracle(n_bits, marked)
    diffuser = build_diffuser(n_bits)
    for _ in range(iterations):
        qc.append(oracle.to_gate(), range(n_bits))
        qc.append(diffuser.to_gate(), range(n_bits))

    qc.measure(range(n_bits), range(n_bits))
    qc = qc.decompose(reps=3)

    sim = AerSimulator()
    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()
    return counts


def main():
    n_bits = 3
    divisor = 3

    # Classical ground truth, computed here from first principles.
    multiples = classical_search(n_bits, divisor)
    print(f"Classical (first-principles) search: n in [0,{2**n_bits}) with "
          f"n % {divisor} == 0 -> {multiples}")

    # Grover search targets a single marked state; use the first nonzero
    # multiple of the divisor as the search target (3, i.e. binary '011').
    target = next(m for m in multiples if m != 0)
    print(f"Grover search target (single marked item): {target} "
          f"(binary {format(target, f'0{n_bits}b')})")

    counts = run_grover(n_bits, target)
    # Bit order from Qiskit measurement is little-endian string; convert.
    winner = max(counts, key=counts.get)
    # Qiskit's measured bitstring already places qubit (n_bits-1) as the
    # most-significant character, matching our little-endian qubit-index
    # convention (qubit i <-> bit i of the target integer) -- no reversal
    # needed.
    winner_int = int(winner, 2)
    total_shots = sum(counts.values())
    winner_fraction = counts[winner] / total_shots

    print(f"Quantum result: most frequent measured state = {winner_int} "
          f"(fraction {winner_fraction:.3f} of {total_shots} shots)")
    print(f"Full counts: {counts}")

    quantum_matches_classical = (winner_int == target) and (winner_fraction > 0.5)

    print()
    if quantum_matches_classical:
        print("PASS: Grover search result matches classical computation "
              "for the toy divisibility-search instance.")
    else:
        print("FAIL: Grover search result does NOT match classical "
              "computation.")

    print()
    print("NOTE: Erdos problem #106 has no OEIS id (oeis: ['N/A']) in the "
          "source data, so this PASS/FAIL verifies only the toy Grover "
          "instance above, not any property of problem #106 itself. See "
          "module docstring for the full explanation.")

    return quantum_matches_classical


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
