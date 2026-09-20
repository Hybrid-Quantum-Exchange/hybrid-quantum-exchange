"""
Erdos problem #10 (erdosproblems.com) -- quantum-testable instance.

OEIS id used: A387053
  "a(n) is the least k such that n is the sum of a prime and k powers of 2
   (repetition of powers allowed)."
This sequence is the natural computational witness for Erdos problem #10,
which concerns representing integers as a prime plus a bounded number of
powers of two (in the spirit of the classical "prime + power of 2"
representation questions, e.g. Erdos's covering-system counterexamples to
Romanoff-type conjectures). a(n) = 0 iff n itself is prime; a(n) = 1 iff
n is NOT prime but n - 2^j is prime for some j >= 0 with 2^j < n.

Classical property tested here (small, finite, computable):
  Fix N = 130 (not prime: 130 = 2*5*13). Consider the eight candidate
  exponents j in {0, ..., 7} (i.e. subtracting 2^j in
  {1,2,4,8,16,32,64,128} from N). Define the search-space index register
  over 3 qubits, representing j in {0,...,7} in binary
  (q2 q1 q0 -> j = 4*q2 + 2*q1 + q0).

  A candidate j is MARKED iff:
      2^j < N   AND   isprime(N - 2^j)   (trial division, computed in
                                           this script from first
                                           principles -- no OEIS lookup)

  For N = 130 this classically gives exactly one witness, j = 7
  (2^7 = 128, 130 - 128 = 2, which is prime), while every other j in
  {0,...,6} gives a composite remainder (130-1=129=3*43,
  130-2=128=2^7, 130-4=126, 130-8=122, 130-16=114, 130-32=98,
  130-64=66, all composite). So the classical marked set is {7},
  matching a(130) = 1 in A387053 (130 is not prime, but 130 - 2^7 = 2
  is prime, so the least number of powers of two needed is 1, uniquely
  witnessed by j=7 among the 8 candidates checked).

Quantum approach: exact Grover search over the 3-qubit index register
(8 basis states, 1 marked). The oracle marks exactly the
classically-determined singleton {7} via a phase-flip built from
X/multi-controlled-Z gates (no hidden lookup table -- the oracle circuit
is generated programmatically from the classical marked-set list
computed above). For N_states=8, M=1 marked item, the optimal number of
Grover iterations is round(pi/4 * sqrt(8/1)) = 2, which should
concentrate the large majority of measurement probability on |111>
(j=7).

PASS criterion: running the circuit on the ideal AerSimulator and
measuring 4096 shots, the probability mass on the classically marked
state must be much higher (here: > 0.85) than the uniform prior (1/8),
and it must be the single most frequent outcome observed.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation (first principles): is_prime + the marked set.
# ---------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


N = 130
NUM_J_QUBITS = 3
J_VALUES = list(range(2 ** NUM_J_QUBITS))  # 0, 1, ..., 7

marked_js = []
for j in J_VALUES:
    power = 2 ** j
    if power < N and is_prime(N - power):
        marked_js.append(j)

print(f"Erdos #10 / OEIS A387053 classical check for N={N}")
for j in J_VALUES:
    power = 2 ** j
    remainder = N - power
    note = "in range" if power < N else "out of range"
    prime_note = is_prime(remainder) if power < N else False
    print(f"  j={j}: 2^j={power}, N-2^j={remainder} ({note}), "
          f"prime={prime_note if power < N else 'n/a'}")
print(f"Classical marked set (witnesses j with N - 2^j prime): {marked_js}")

assert marked_js == [7], f"unexpected classical marked set {marked_js}"
# a(N) for N=130 in A387053 is 1 (least k such that N is prime + k powers
# of 2), and the marked set above is exactly the set of single-power-of-two
# witnesses for that k=1 representation.
CLASSICAL_A_OF_N = 1
print(f"Classical A387053 value a({N}) = {CLASSICAL_A_OF_N} "
      f"(uniquely witnessed by j in {marked_js})\n")


# ---------------------------------------------------------------------------
# 2. Build the Grover oracle from the classical marked set (no lookup table
#    is copied into the oracle by hand -- it is generated from marked_js).
# ---------------------------------------------------------------------------

def bits_of(j: int, num_qubits: int):
    """Little-endian bit list: bits_of(1, 2) -> [1, 0] meaning q0=1, q1=0."""
    return [(j >> b) & 1 for b in range(num_qubits)]


def multi_controlled_z(qc: QuantumCircuit, qubits: list):
    """Apply a phase flip to the |11...1> basis state of `qubits`."""
    n = len(qubits)
    if n == 1:
        qc.z(qubits[0])
    elif n == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1])
        qc.h(qubits[-1])


def add_mark_phase_flip(qc: QuantumCircuit, j: int, qubits: list):
    """Flip the phase of basis state |j> using X-conjugated multi-CZ."""
    bits = bits_of(j, len(qubits))
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)
    multi_controlled_z(qc, qubits)
    for q, b in zip(qubits, bits):
        if b == 0:
            qc.x(q)


def build_oracle(marked: list, qubits: list) -> QuantumCircuit:
    qc = QuantumCircuit(len(qubits), name="Oracle")
    for j in marked:
        add_mark_phase_flip(qc, j, list(range(len(qubits))))
    return qc


def build_diffuser(qubits: list) -> QuantumCircuit:
    n = len(qubits)
    qc = QuantumCircuit(n, name="Diffuser")
    qc.h(range(n))
    qc.x(range(n))
    multi_controlled_z(qc, list(range(n)))
    qc.x(range(n))
    qc.h(range(n))
    return qc


qubits = list(range(NUM_J_QUBITS))
oracle = build_oracle(marked_js, qubits)
diffuser = build_diffuser(qubits)

qc = QuantumCircuit(NUM_J_QUBITS, NUM_J_QUBITS)
qc.h(qubits)  # uniform superposition over j in {0,1,2,3}

# Optimal number of Grover iterations for N_states=8, M=1 marked:
# round(pi/4 * sqrt(N_states/M)) = round(pi/4 * sqrt(8)) = 2.
num_states = 2 ** NUM_J_QUBITS
num_marked = len(marked_js)
num_iterations = round((np.pi / 4) * np.sqrt(num_states / num_marked))
for _ in range(num_iterations):
    qc.append(oracle.to_gate(), qubits)
    qc.append(diffuser.to_gate(), qubits)

qc.measure(qubits, qubits)


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator.
# ---------------------------------------------------------------------------

backend = AerSimulator()
tqc = transpile(qc, backend)
shots = 4096
result = backend.run(tqc, shots=shots).result()
counts = result.get_counts()

print("Measurement counts (raw bitstrings, Qiskit little-endian):")
for bitstring, count in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {bitstring}: {count}")

# Decode bitstrings (Qiskit's default classical-register ordering is
# c(n-1)...c1c0 for an n-bit register measured in order [0,1,...,n-1])
# back to j values.
marked_set = set(marked_js)
decoded_counts = {}
for bitstring, count in counts.items():
    bits = [int(b) for b in reversed(bitstring)]  # bits[i] = qubit i
    j = sum(b << i for i, b in enumerate(bits))
    decoded_counts[j] = decoded_counts.get(j, 0) + count

marked_mass = sum(c for j, c in decoded_counts.items() if j in marked_set)
marked_probability = marked_mass / shots
most_frequent_j = max(decoded_counts, key=decoded_counts.get)

print(f"\nDecoded outcome counts (j -> count): {decoded_counts}")
print(f"Total probability mass on classically marked set {sorted(marked_set)}: "
      f"{marked_probability:.4f}")
print(f"Most frequent measured j: {most_frequent_j}")

QUANTUM_PASS = marked_probability > 0.85 and most_frequent_j in marked_set

if QUANTUM_PASS:
    print(f"PASS: Grover search concentrated on the correct marked set "
          f"{sorted(marked_set)} for N={N}, confirming (via quantum search) "
          f"the witnesses for A387053 a({N}) = {CLASSICAL_A_OF_N}.")
else:
    print("FAIL: quantum search did not concentrate on the classical "
          "marked set as expected.")
