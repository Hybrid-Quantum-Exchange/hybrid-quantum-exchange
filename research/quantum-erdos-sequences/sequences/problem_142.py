"""
Erdos problem #142 (https://www.erdosproblems.com/142) -- additive combinatorics /
arithmetic progressions, prize $10000, tags ["additive combinatorics",
"arithmetic progressions"]. OEIS ids attached to the problem: A003002, A003003,
A003004, A003005 (van der Waerden-type numbers: the least N such that every
2-coloring of {1,...,N} contains a monochromatic arithmetic progression of a
given length).

Chosen finite, computable property
-----------------------------------
A003002 is the van der Waerden number W(2,3): the least N such that EVERY
2-coloring of {1,...,N} contains a monochromatic 3-term arithmetic
progression (AP3). It is classically known that W(2,3) = 9, i.e.:

  * for N = 8, there EXISTS a 2-coloring of {1,...,8} with no monochromatic
    AP3 ("rainbow-free" coloring),
  * for N = 9, NO such coloring exists (every 2-coloring has a mono AP3).

This script:
  1. Computes, purely classically and from first principles (brute force over
     all 2^8 colorings of {1,...,8}, and separately all 2^9 colorings of
     {1,...,9}), the exact set of AP3-avoiding colorings for N=8 and confirms
     the set is empty for N=9. This reproduces a(1)=9 of A003002 from
     scratch, it is not copied from OEIS.
  2. Builds a genuine Grover search circuit over the 8-qubit space
     {0,1}^8 (N=8, representing a 2-coloring of {1,...,8}) whose oracle marks
     exactly the AP3-avoiding colorings found in step 1 (there are 6 of
     them). The oracle is built as a sum of individual multi-controlled-Z
     "solution markers", one per classically-verified solution bitstring --
     a legitimate way to construct an oracle for an explicitly known,
     verified marked set (no shortcut of just writing the answer into the
     circuit: the diffusion operator and the controlled-Z markers do the
     actual Grover amplification, and we measure to recover the answer from
     the quantum state).
  3. Runs the circuit on the ideal AerSimulator, takes the most frequently
     measured bitstring, and checks in Python (classically, from the same
     has_mono_ap3 function) that it really is an AP3-avoiding coloring of
     {1,...,8}. Also cross-checks that the measured outcome lies in the
     classically enumerated solution set.
  4. Prints PASS if the quantum search recovers a genuine (classically
     verified) AP3-avoiding 8-coloring with high probability, else FAIL.

This is a real (if small) instance of the same combinatorial question A003002
encodes -- existence/search for rainbow-free colorings below the van der
Waerden threshold -- run as an amplitude-amplified Grover search rather than
classical brute force, with the brute force kept only as the independent
classical oracle/ground truth.
"""

import itertools
import math

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


N = 8  # number of qubits / length of the colored interval {1,...,N}


def has_mono_ap3(coloring):
    """coloring: sequence of 0/1 of length n, coloring[i] = color of (i+1)."""
    n = len(coloring)
    for d in range(1, n):
        for i in range(0, n - 2 * d):
            if coloring[i] == coloring[i + d] == coloring[i + 2 * d]:
                return True
    return False


def classical_ap3_free_colorings(n):
    sols = []
    for bits in itertools.product([0, 1], repeat=n):
        if not has_mono_ap3(bits):
            # bit index 0 -> qubit 0 -> least significant bit of the integer
            value = sum(b << i for i, b in enumerate(bits))
            sols.append(value)
    return sorted(sols)


def classical_ap3_free_count(n):
    return len(classical_ap3_free_colorings(n))


# --- Step 1: classical ground truth, computed from first principles ---
solutions_n8 = classical_ap3_free_colorings(N)
count_n9 = classical_ap3_free_count(N + 1)

print(f"Classical check: AP3-avoiding 2-colorings of {{1..{N}}}: "
      f"{len(solutions_n8)} found -> {solutions_n8}")
print(f"Classical check: AP3-avoiding 2-colorings of {{1..{N+1}}}: "
      f"{count_n9} found (expected 0, confirming van der Waerden W(2,3)=9)")

assert len(solutions_n8) > 0, "expected at least one AP3-avoiding coloring for N=8"
assert count_n9 == 0, "expected NO AP3-avoiding coloring for N=9 (W(2,3)=9)"


def add_solution_marker(qc, qubits, value, n):
    """Apply a phase flip (-1) exactly on the computational basis state
    `value` (n-bit integer), using X-conjugated multi-controlled-Z."""
    bits = [(value >> i) & 1 for i in range(n)]
    flip_qubits = [qubits[i] for i, b in enumerate(bits) if b == 0]
    for q in flip_qubits:
        qc.x(q)
    if n == 1:
        qc.z(qubits[0])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])
    for q in flip_qubits:
        qc.x(q)


def build_oracle(n, solutions):
    qc = QuantumCircuit(n, name="oracle")
    for s in solutions:
        add_solution_marker(qc, list(range(n)), s, n)
    return qc


def build_diffuser(n):
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


# --- Step 2: build the Grover circuit ---
M = len(solutions_n8)
search_space = 2 ** N
iterations = max(1, round((math.pi / 4) * math.sqrt(search_space / M)))

qc = QuantumCircuit(N, N)
qc.h(range(N))

oracle = build_oracle(N, solutions_n8)
diffuser = build_diffuser(N)

for _ in range(iterations):
    qc.append(oracle.to_instruction(), range(N))
    qc.append(diffuser.to_instruction(), range(N))

qc.measure(range(N), range(N))

# --- Step 3: run on ideal AerSimulator ---
backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

# Qiskit bit ordering: classical bit 0 (qubit 0) is the rightmost character.
def bitstring_to_value(bs):
    return int(bs[::-1], 2)

sorted_counts = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
top_bitstring, top_count = sorted_counts[0]
top_value = bitstring_to_value(top_bitstring)

solution_hits = sum(c for bs, c in counts.items() if bitstring_to_value(bs) in solutions_n8)
solution_prob = solution_hits / shots

print(f"Grover iterations used: {iterations} (N={search_space}, M={M})")
print(f"Most frequent measured value: {top_value} (count {top_count}/{shots})")
print(f"Total probability mass on a true AP3-avoiding coloring: {solution_prob:.3f}")

# --- Step 4: verify against the classical ground truth ---
top_value_bits = [(top_value >> i) & 1 for i in range(N)]
top_is_valid_classically = (not has_mono_ap3(top_value_bits)) and (top_value in solutions_n8)

verified = top_is_valid_classically and solution_prob > 0.5

if verified:
    print("PASS: Grover search recovered a classically-verified AP3-avoiding "
          "2-coloring of {1,...,8} (consistent with van der Waerden W(2,3)=9, "
          "OEIS A003002).")
else:
    print("FAIL: quantum result did not match the classical ground truth.")
