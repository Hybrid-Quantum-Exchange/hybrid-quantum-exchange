"""
Erdos problem #586 (erdosproblems.com) — quantum-testable instance.

Problem #586 is about *covering systems* of congruences (tags in
data/problems.yaml: ["number theory", "covering systems"]; status:
"disproved"; prize: "no"). The problems.yaml entry for this problem lists
``oeis: ["N/A"]`` — there is no OEIS sequence attached to problem #586, so
this script does NOT test an OEIS term. Per the task instructions, when no
OEIS id exists, the honest fallback is to build a genuine small, finite,
computable property drawn from the problem's own subject matter (covering
systems) and verify it classically before testing it on a real quantum
circuit. That is what this script does.

Classical property being tested
--------------------------------
A "covering system" is a finite set of congruences (a_i mod m_i) such that
every integer satisfies at least one of them. Erdos's own classical example
of a covering system is:

    0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12      (covers all integers)

To get a *finite, small, and non-trivial* search instance (rather than one
whose answer set is empty), this script drops the "5 mod 6" congruence,
leaving:

    C = { 0 mod 2, 0 mod 3, 1 mod 4, 7 mod 12 }

and asks: which residues x in [0, 11] (mod 12, so all moduli's period
divides the search range) are NOT covered by C? This is a small, finite,
exactly-computable set-membership property with genuine mathematical
content connected directly to problem #586's subject (covering systems).

The classical answer is computed here from first principles (brute force,
no external lookup): x=11 is the unique element of [0,11] not covered by C
(0,2,3,4,6,8,9,10 are hit by "0 mod 2" or "0 mod 3"; 1,5 are hit by
"1 mod 4"; 7 is hit by "7 mod 12"; 11 hits none of the four congruences).

Quantum circuit
----------------
We use Grover's algorithm on 4 qubits (search space N = 16, of which the
valid instance domain is [0, 11]) with a diagonal oracle that phase-flips
exactly the classical uncovered set ({11}, a single marked item encoded as
bitstring '1011'). With one marked item out of N = 16, the optimal number
of Grover iterations is round(pi/4 * sqrt(16/1)) = 3. We run the circuit on
the ideal AerSimulator and check that the most frequently measured
bitstring equals the classically-computed uncovered residue.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


# ---------------------------------------------------------------------------
# 1. Classical computation of the covering-system property (ground truth).
# ---------------------------------------------------------------------------

def covers(x, a, m):
    """True if integer x satisfies the congruence x = a (mod m)."""
    return x % m == a


def compute_uncovered_residues(congruences, domain):
    """Brute-force: residues in `domain` not hit by any congruence in
    `congruences` (a list of (a, m) pairs)."""
    uncovered = []
    for x in domain:
        if not any(covers(x, a, m) for (a, m) in congruences):
            uncovered.append(x)
    return uncovered


# Erdos's classical covering system with "5 mod 6" removed.
CONGRUENCES = [(0, 2), (0, 3), (1, 4), (7, 12)]
DOMAIN = list(range(12))  # x = 0..11, since lcm(2,3,4,12) = 12

uncovered = compute_uncovered_residues(CONGRUENCES, DOMAIN)
assert uncovered == [11], f"expected classical uncovered set [11], got {uncovered}"

MARKED_VALUE = uncovered[0]  # 11
N_QUBITS = 4                 # 2**4 = 16 >= 12, one bitstring per marked value
MARKED_BITSTRING = format(MARKED_VALUE, f"0{N_QUBITS}b")  # '1011'

print(f"Classical check: congruences {CONGRUENCES} over domain {DOMAIN}")
print(f"Classical uncovered residue(s) mod 12: {uncovered}")
print(f"Marked bitstring for Grover oracle: {MARKED_BITSTRING} (value {MARKED_VALUE})")


# ---------------------------------------------------------------------------
# 2. Grover's algorithm: oracle marks the classically-uncovered residue.
# ---------------------------------------------------------------------------

def build_oracle(n_qubits, marked_bitstring):
    """Phase-flip the single computational basis state |marked_bitstring>."""
    qc = QuantumCircuit(n_qubits, name="oracle")
    # Flip qubits where the marked bit is 0, so the all-ones pattern lines
    # up with a multi-controlled Z, then flip back.
    zero_positions = [i for i, b in enumerate(reversed(marked_bitstring)) if b == "0"]
    for i in zero_positions:
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in zero_positions:
        qc.x(i)
    return qc


def build_diffuser(n_qubits):
    """Standard Grover diffusion operator (inversion about the mean)."""
    qc = QuantumCircuit(n_qubits, name="diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def grover_iterations(n_qubits, num_marked):
    N = 2 ** n_qubits
    return max(1, round((np.pi / 4) * np.sqrt(N / num_marked)))


n_iterations = grover_iterations(N_QUBITS, num_marked=1)
print(f"Running Grover with {n_iterations} iteration(s) over {2**N_QUBITS} states")

qc = QuantumCircuit(N_QUBITS, N_QUBITS)
qc.h(range(N_QUBITS))

oracle = build_oracle(N_QUBITS, MARKED_BITSTRING)
diffuser = build_diffuser(N_QUBITS)

for _ in range(n_iterations):
    qc.append(oracle.to_gate(), range(N_QUBITS))
    qc.append(diffuser.to_gate(), range(N_QUBITS))

qc.measure(range(N_QUBITS), range(N_QUBITS))
qc = qc.decompose().decompose()


# ---------------------------------------------------------------------------
# 3. Run on the ideal AerSimulator and compare to the classical answer.
# ---------------------------------------------------------------------------

backend = AerSimulator()
job = backend.run(qc, shots=2048)
result = job.result()
counts = result.get_counts()

# Qiskit's classical register bit order is c[N-1]...c[0]; our bitstrings
# above were built MSB-first with qubit 0 as the least-significant bit, so
# Qiskit's returned bitstrings (also MSB..LSB of the register, matching
# qubit N-1..0) are directly comparable to MARKED_BITSTRING.
most_common_bitstring, most_common_count = max(counts.items(), key=lambda kv: kv[1])
most_common_value = int(most_common_bitstring, 2)

print(f"Measurement counts: {counts}")
print(f"Most frequent outcome: {most_common_bitstring} (value {most_common_value}), "
      f"{most_common_count}/{sum(counts.values())} shots")

quantum_matches_classical = (most_common_value == MARKED_VALUE)

if quantum_matches_classical:
    print("PASS")
else:
    print("FAIL")
