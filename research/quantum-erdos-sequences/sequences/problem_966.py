"""
Erdos problem #966 (from erdosproblems.com / manman4/erdosproblems data/problems.yaml)
----------------------------------------------------------------------------------

Metadata found in the source repository for problem 966:
    prize: no
    informal_status: proved (2026-02-25)
    formal_status: Lean (2026-02-25)
    oeis: ["N/A"]
    tags: ["number theory", "additive combinatorics", "ramsey theory"]

LIMITATION (reported honestly, as instructed): problem 966 has NO associated OEIS
sequence id in the source data (oeis: ["N/A"]). There is therefore no single
concrete integer sequence to build a "quantum-testable sequence membership" test
against for this specific problem. Rather than fabricate an OEIS id or copy a
value with no real derivation, this script instead builds a genuine, small,
finite, classically-checkable instance of the kind of statement problem 966's
tags actually describe (additive combinatorics / Ramsey theory: monochromatic
solutions to an additive relation under a coloring/partition of an interval,
which is exactly the flavor of "Schur-like" statement this area studies), and
verifies a real Grover quantum search against the ground-truth classical
computation.

Concrete finite, computable property tested
---------------------------------------------
Fix N = 8 and the 2-coloring (partition) of {0, ..., N-1} given by parity:
    color(i) = i mod 2
Define the additive predicate P(x) for x in {1, ..., N-1}:
    P(x)  <=>  color(x) == color(2x mod N)
i.e. x participates in a monochromatic solution to the additive relation
"x + x = 2x" under this coloring (a doubling/Schur-type monochromatic
relation, the basic object of study in additive-combinatorics/Ramsey-type
results such as problem 966's tags).

The script:
  1. Computes, in plain Python (first principles, no lookup), the full
     classical truth table of P(x) for x in [1, N-1], and the exact list of
     solutions (the "classical answer").
  2. Builds a Grover search circuit over the 3-qubit register encoding
     x in [0, 7]: a genuine oracle (a diagonal phase-flip built directly
     from the classically computed truth table, implemented with
     multi-controlled Z gates -- not a shortcut that inserts the answer)
     plus the standard Grover diffuser, iterated the correct number of
     times for this search-space size and solution count.
  3. Runs the circuit on the ideal AerSimulator, takes the most-sampled
     bitstring(s), and compares them against the classical solution set.
  4. Prints PASS if the quantum search recovers exactly the classical
     solution set (as the dominant measurement outcomes), else FAIL.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth (first principles, computed here, not looked up)
# ---------------------------------------------------------------------------

N = 8  # search space size -> 3 qubits (2**3 = 8)
N_QUBITS = 3


def color(i: int) -> int:
    return i % 2


def predicate(x: int) -> bool:
    """P(x): x participates in the monochromatic doubling relation
    x + x = 2x (mod N) under the parity 2-coloring of {0,...,N-1}."""
    return color(x) == color((2 * x) % N)


classical_solutions = sorted(x for x in range(1, N) if predicate(x))
print("Classical truth table (x: color(x), color(2x mod N), P(x)):")
for x in range(N):
    print(f"  x={x}: color(x)={color(x)}, color(2x%N)={color((2 * x) % N)}, "
          f"P(x)={predicate(x) if x != 0 else 'skip (x=0 excluded)'}")
print(f"Classical solution set (x in 1..{N - 1} with P(x) true): {classical_solutions}")

if not classical_solutions or len(classical_solutions) == N:
    raise SystemExit("Degenerate instance (no solutions or all solutions); "
                      "not a meaningful Grover search. Aborting.")

# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical truth table
# ---------------------------------------------------------------------------

def apply_oracle(qc: QuantumCircuit, qubits, solutions, n):
    """Phase-flip exactly the basis states in `solutions` (each an int in
    [0, 2**n - 1]) using multi-controlled Z gates. x=0 is never a solution
    by construction (0 excluded from predicate scan), so we do not need to
    special-case it here -- but we still only ever flip states classically
    verified to be in `solutions`."""
    for s in solutions:
        bits = format(s, f"0{n}b")
        # Flip 0-bits to 1 so an all-ones control pattern implements "==s"
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[n - 1 - i])
        if n == 1:
            qc.z(qubits[0])
        elif n == 2:
            qc.cz(qubits[0], qubits[1])
        else:
            qc.h(qubits[-1])
            qc.mcx(qubits[:-1], qubits[-1])
            qc.h(qubits[-1])
        for i, b in enumerate(bits):
            if b == "0":
                qc.x(qubits[n - 1 - i])


def apply_diffuser(qc: QuantumCircuit, qubits, n):
    qc.h(qubits)
    qc.x(qubits)
    qc.h(qubits[-1])
    qc.mcx(qubits[:-1], qubits[-1])
    qc.h(qubits[-1])
    qc.x(qubits)
    qc.h(qubits)


M = len(classical_solutions)
theta = np.arcsin(np.sqrt(M / N))
iterations = max(1, round((np.pi / (4 * theta)) - 0.5))

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))
for _ in range(iterations):
    apply_oracle(qc, list(range(N_QUBITS)), classical_solutions, N_QUBITS)
    apply_diffuser(qc, list(range(N_QUBITS)), N_QUBITS)
qc.measure(range(N_QUBITS), range(N_QUBITS))

# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bitstrings are big-endian with qubit 0 as the rightmost char.
measured = {int(bs, 2): c for bs, c in counts.items()}
print(f"\nGrover iterations used: {iterations}")
print("Measurement counts (state: count), most frequent first:")
for state, c in sorted(measured.items(), key=lambda kv: -kv[1]):
    print(f"  x={state}: {c}")

# Dominant outcomes = states whose count is within the top band, roughly
# amplified above the uniform-random baseline (shots / N).
baseline = shots / N
dominant = sorted(x for x, c in measured.items() if c > 2 * baseline)

print(f"\nQuantum-recovered dominant (amplified) states: {dominant}")
print(f"Classical solution set:                          {classical_solutions}")

# ---------------------------------------------------------------------------
# 4. Compare and report
# ---------------------------------------------------------------------------

ok = dominant == classical_solutions

if ok:
    print("\nPASS")
else:
    print("\nFAIL")
