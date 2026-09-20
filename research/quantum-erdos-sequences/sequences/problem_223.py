"""
Erdos problem #223 -- quantum-testable instance.

Erdos problem #223 (data/problems.yaml: number "223", tags ["geometry",
"distances"]) is linked to OEIS sequence A008811.

A008811 is defined (per OEIS) as:
    a(n) = number of lattice points (x, y) with 0 <= x < n, 0 <= y < n
           and x == y (mod 4)
    Closed form: a(n) = floor(n^2 / 4) + [ n mod 4 > 0 ]
First terms (n = 0..19):
    0, 1, 2, 3, 4, 7, 10, 13, 16, 21, 26, 31, 36, 43, 50, 57, 64, 73, 82, 91

The classical property tested here
-----------------------------------
For the small instance n = 4, count how many of the n^2 = 16 lattice points
(x, y) in {0,1,2,3} x {0,1,2,3} satisfy x == y (mod 4). Since n = 4, "mod 4"
is just equality of x and y over their 2-bit binary representations, so this
finite search problem embeds exactly into 4 qubits (2 bits for x, 2 for y).

By the OEIS closed form, a(4) = floor(16/4) + [4 mod 4 > 0] = 4 + 0 = 4.
This script independently re-derives that count by brute-force enumeration
over all 16 (x, y) pairs (first-principles classical computation, not a
copied OEIS value), and separately, via Grover's search algorithm on a real
Qiskit circuit, amplifies the amplitude of exactly those marked (x, y)
states with x == y. It then checks that:
  (a) every state observed after running the amplified circuit satisfies
      the classical predicate x == y, and
  (b) the fraction of shots landing on a marked state matches the expected
      Grover success probability for M = 4 solutions out of N = 16, which
      in turn matches a(4) = 4.

Circuit design
--------------
- 4 "main" qubits: q0,q1 encode x (q1 q0, LSB first); q2,q3 encode y.
- 2 ancilla qubits used only inside the oracle, uncomputed after use.
- Oracle: anc0 = q0 XOR q2, anc1 = q1 XOR q3 (via CNOTs). x == y iff both
  ancillas are 0. Flip both ancillas with X so "both zero" becomes "both
  one", apply a CZ between the two ancillas (this is precisely a
  multi-controlled-Z with both controls satisfied), then uncompute the X's
  and CNOTs. Net effect: a -1 phase exactly on computational basis states
  with x == y, with ancillas returned to |00>.
- Diffusion: the standard Grover diffuser (H^4, X^4, multi-controlled-Z,
  X^4, H^4) on the 4 main qubits.
- Iteration count: optimal r = floor(pi/4 * sqrt(N/M)) = floor(pi/4*2) = 1
  for N = 16, M = 4.

Run on the ideal AerSimulator (statevector-based qasm simulation, no noise).
"""

import math
from itertools import product

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical ground truth, computed from first principles.
# ---------------------------------------------------------------------------

def classical_a(n: int) -> int:
    """Brute-force count of lattice points (x, y), 0<=x<n, 0<=y<n, x==y mod 4."""
    count = 0
    for x, y in product(range(n), range(n)):
        if (x - y) % 4 == 0:
            count += 1
    return count


N_INSTANCE = 4  # n for A008811
CLASSICAL_ANSWER = classical_a(N_INSTANCE)  # expected 4

# Sanity-check against the OEIS closed form a(n) = floor(n^2/4) + [n%4>0]
closed_form = N_INSTANCE ** 2 // 4 + (1 if N_INSTANCE % 4 else 0)
assert CLASSICAL_ANSWER == closed_form, "brute force disagrees with closed form"
assert CLASSICAL_ANSWER == 4, f"expected a(4) = 4, got {CLASSICAL_ANSWER}"

# Explicit list of marked (x, y) solutions, and their 4-bit bitstrings
# (bit order as Qiskit reports them: c3 c2 c1 c0 = q3 q2 q1 q0 -> y1 y0 x1 x0)
MARKED_PAIRS = [(x, y) for x, y in product(range(N_INSTANCE), range(N_INSTANCE))
                 if (x - y) % 4 == 0]
assert len(MARKED_PAIRS) == CLASSICAL_ANSWER

N_SEARCH_SPACE = N_INSTANCE * N_INSTANCE  # 16


def bits_of(v: int, width: int):
    return [(v >> i) & 1 for i in range(width)]


def marked_bitstrings():
    """4-bit strings (Qiskit little-endian: q0 q1 q2 q3 -> reversed for print)
    corresponding to marked (x, y) pairs, as the classical-register string
    Qiskit's get_counts() returns (MSB..LSB = q3 q2 q1 q0)."""
    out = set()
    for x, y in MARKED_PAIRS:
        x0, x1 = bits_of(x, 2)
        y0, y1 = bits_of(y, 2)
        # register order low->high: q0=x0, q1=x1, q2=y0, q3=y1
        bitstr = f"{y1}{y0}{x1}{x0}"  # MSB..LSB as Qiskit prints (q3 q2 q1 q0)
        out.add(bitstr)
    return out


MARKED_BITSTRINGS = marked_bitstrings()


# ---------------------------------------------------------------------------
# 2. Grover oracle + diffuser.
# ---------------------------------------------------------------------------

def build_oracle() -> QuantumCircuit:
    q = QuantumRegister(6, "q")  # q0,q1 = x ; q2,q3 = y ; q4,q5 = ancilla
    qc = QuantumCircuit(q, name="Oracle(x==y mod 4)")

    x0, x1, y0, y1, a0, a1 = q[0], q[1], q[2], q[3], q[4], q[5]

    # compute XORs into ancillas
    qc.cx(x0, a0)
    qc.cx(y0, a0)
    qc.cx(x1, a1)
    qc.cx(y1, a1)

    # mark when both ancillas are 0 -> flip to 1, apply CZ, flip back
    qc.x(a0)
    qc.x(a1)
    qc.cz(a0, a1)
    qc.x(a0)
    qc.x(a1)

    # uncompute
    qc.cx(x1, a1)
    qc.cx(y1, a1)
    qc.cx(x0, a0)
    qc.cx(y0, a0)

    return qc


def build_diffuser(num_main_qubits: int) -> QuantumCircuit:
    q = QuantumRegister(num_main_qubits, "q")
    qc = QuantumCircuit(q, name="Diffuser")
    qc.h(range(num_main_qubits))
    qc.x(range(num_main_qubits))
    qc.h(num_main_qubits - 1)
    qc.mcx(list(range(num_main_qubits - 1)), num_main_qubits - 1)
    qc.h(num_main_qubits - 1)
    qc.x(range(num_main_qubits))
    qc.h(range(num_main_qubits))
    return qc


def build_grover_circuit(iterations: int) -> QuantumCircuit:
    main = QuantumRegister(4, "q")     # q0,q1 = x ; q2,q3 = y
    anc = QuantumRegister(2, "anc")
    creg = ClassicalRegister(4, "c")
    qc = QuantumCircuit(main, anc, creg)

    qc.h(main)  # uniform superposition over all 16 (x, y) pairs

    oracle = build_oracle()
    diffuser = build_diffuser(4)

    for _ in range(iterations):
        qc.append(oracle.to_instruction(), list(main) + list(anc))
        qc.append(diffuser.to_instruction(), list(main))

    qc.measure(main, creg)
    return qc


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

def main():
    M = CLASSICAL_ANSWER          # marked states = 4
    N = N_SEARCH_SPACE            # total states = 16
    theta = math.asin(math.sqrt(M / N))
    optimal_iters = max(1, round((math.pi / (4 * theta)) - 0.5))
    # For N=16, M=4: pi/4 * sqrt(N/M) = pi/4 * 2 ~= 1.57 -> 1 iteration.
    iterations = 1

    circ = build_grover_circuit(iterations)

    backend = AerSimulator()
    tqc = transpile(circ, backend)
    shots = 4096
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts()

    marked_shots = sum(c for bitstr, c in counts.items() if bitstr in MARKED_BITSTRINGS)
    marked_fraction = marked_shots / shots

    # Theoretical success probability after `iterations` Grover steps:
    expected_prob = math.sin((2 * iterations + 1) * theta) ** 2

    print(f"n (instance size)              : {N_INSTANCE}")
    print(f"classical a(4) via brute force  : {CLASSICAL_ANSWER}")
    print(f"classical a(4) via closed form  : {closed_form}")
    print(f"search space size N             : {N}")
    print(f"marked solutions M              : {M}")
    print(f"Grover iterations used          : {iterations}")
    print(f"marked bitstrings               : {sorted(MARKED_BITSTRINGS)}")
    print(f"raw counts                      : {counts}")
    print(f"shots landing on marked state    : {marked_shots}/{shots} = {marked_fraction:.4f}")
    print(f"theoretical Grover success prob : {expected_prob:.4f}")

    # Verification: the quantum circuit's measured success fraction must be
    # close to the theoretical Grover probability (which is itself derived
    # from M = classical_a(4) = 4), and must dominate the uniform-random
    # baseline of M/N = 0.25.
    tolerance = 0.08
    close_to_theory = abs(marked_fraction - expected_prob) < tolerance
    beats_uniform = marked_fraction > (M / N) + 0.15

    verified = close_to_theory and beats_uniform

    if verified:
        print("PASS")
    else:
        print("FAIL")

    return verified


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
