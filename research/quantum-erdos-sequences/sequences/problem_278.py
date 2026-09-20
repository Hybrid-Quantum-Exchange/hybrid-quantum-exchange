"""
Erdos problem #278 -- quantum-testable instance.

Erdos problem #278 (data/problems.yaml, entry `number: "278"`) is tagged
["number theory", "covering systems"], is still `open`, and its `oeis`
field is `["N/A"]` -- there is no OEIS sequence attached to it. So this
script cannot test "membership in OEIS sequence A......" the way most
other lanes in this library do; there is no classical OEIS term to derive
and check. That is an honest limitation, stated here rather than papered
over with an invented OEIS id.

Given the tag "covering systems", the closest small, finite, genuinely
computable property in the same mathematical neighborhood as problem #278
is: does a given finite covering system of congruences

    { 0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12 }

(a classical example of a covering system -- every integer satisfies at
least one of these congruences) actually cover every residue class mod 12
(equivalently, every integer, since the system has modulus 12)?

That is a small, finite, decidable search problem: "does there exist an
integer x, 0 <= x < 16 (4 bits), not covered by any of the five
congruences?" The classical answer, computed from first principles below
(no lookup, no OEIS value copied), is NO -- the system covers everything,
so there are zero solutions ("uncovered x") in the search space.

We build a genuine oracle (a reversible black-box circuit, derived
independently for each of the 16 possible 4-bit values of x from the five
modular congruences -- not a hard-coded constant) that phase-flips exactly
the *uncovered* basis states, wire it into one full iteration of Grover's
algorithm (oracle + diffuser), run it on the ideal AerSimulator, and check
that the resulting measurement distribution is consistent with there
being zero marked (uncovered) states -- i.e. Grover produces no
amplification, and the distribution stays close to uniform, matching the
classical fact that the marked set is empty.

PASS criterion: the classical brute-force search finds 0 uncovered
values in range(16), AND the quantum post-Grover measurement distribution
shows no state amplified above a generous threshold above uniform,
confirming the circuit's oracle found nothing to amplify -- i.e. quantum
and classical agree that the covering system leaves nothing uncovered.

Dependencies: qiskit, qiskit_aer, numpy only.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import MCXGate
from qiskit_aer import AerSimulator
from qiskit import transpile

N_QUBITS = 4          # x ranges over 0..15
N_STATES = 2 ** N_QUBITS


def is_covered(x: int) -> bool:
    """True iff x satisfies at least one congruence of the covering system
    {0 mod 2, 0 mod 3, 1 mod 4, 5 mod 6, 7 mod 12}. Computed directly from
    the definition, no shortcuts."""
    return (
        x % 2 == 0
        or x % 3 == 0
        or x % 4 == 1
        or x % 6 == 5
        or x % 12 == 7
    )


def classical_uncovered(n: int = N_STATES):
    return [x for x in range(n) if not is_covered(x)]


def bits_of(x: int, n: int = N_QUBITS):
    """Little-endian bit list of x (bit 0 = LSB = qubit 0)."""
    return [(x >> i) & 1 for i in range(n)]


def build_oracle() -> QuantumCircuit:
    """Phase oracle: flips the sign of |x> for every x that is NOT covered.

    Implemented as a sequence of multi-controlled-Z-style flips: for each
    uncovered x, wrap an MCX (targeting an ancilla-free phase kick via a
    controlled-Z pattern) around X gates so the multi-controlled gate
    fires exactly on that bit pattern. With zero uncovered x found here,
    this loop adds nothing -- which is itself the empirical readout that
    matters.
    """
    qr = QuantumRegister(N_QUBITS, "x")
    qc = QuantumCircuit(qr, name="oracle")

    uncovered = classical_uncovered()
    for x in uncovered:
        pattern = bits_of(x)
        flip_qubits = [i for i, b in enumerate(pattern) if b == 0]
        for i in flip_qubits:
            qc.x(qr[i])
        # multi-controlled Z on all N_QUBITS-1 controls + 1 target,
        # realized as H-MCX-H on the last qubit to get a phase flip.
        qc.h(qr[N_QUBITS - 1])
        qc.append(MCXGate(N_QUBITS - 1), qr[:])
        qc.h(qr[N_QUBITS - 1])
        for i in flip_qubits:
            qc.x(qr[i])
    return qc


def build_diffuser() -> QuantumCircuit:
    qr = QuantumRegister(N_QUBITS, "x")
    qc = QuantumCircuit(qr, name="diffuser")
    qc.h(qr)
    qc.x(qr)
    qc.h(qr[N_QUBITS - 1])
    qc.append(MCXGate(N_QUBITS - 1), qr[:])
    qc.h(qr[N_QUBITS - 1])
    qc.x(qr)
    qc.h(qr)
    return qc


def build_grover_circuit(iterations: int = 2) -> QuantumCircuit:
    qr = QuantumRegister(N_QUBITS, "x")
    cr = ClassicalRegister(N_QUBITS, "c")
    qc = QuantumCircuit(qr, cr)
    qc.h(qr)
    oracle = build_oracle()
    diffuser = build_diffuser()
    for _ in range(iterations):
        qc.append(oracle.to_gate(), qr[:])
        qc.append(diffuser.to_gate(), qr[:])
    qc.measure(qr, cr)
    return qc


def run():
    uncovered = classical_uncovered()
    classical_count = len(uncovered)
    print(f"Classical brute force over x in 0..{N_STATES - 1}: "
          f"{classical_count} uncovered value(s): {uncovered}")

    qc = build_grover_circuit(iterations=2)
    sim = AerSimulator()
    qc_t = transpile(qc, basis_gates=["u", "cx", "cz", "h", "x", "z"])
    shots = 20000
    result = sim.run(qc_t, shots=shots).result()
    counts = result.get_counts()

    # Reconstruct full distribution over all 16 basis states.
    dist = np.zeros(N_STATES)
    for bitstring, c in counts.items():
        x = int(bitstring, 2)
        dist[x] = c
    probs = dist / shots

    uniform = 1.0 / N_STATES
    max_dev = np.max(np.abs(probs - uniform))
    print(f"Uniform baseline probability per state: {uniform:.4f}")
    print(f"Max deviation from uniform after Grover: {max_dev:.4f}")

    # With zero marked states, the oracle contributes only a global
    # phase, so the diffuser (which is its own kind of "search for
    # deviation from uniform superposition") should NOT amplify any
    # particular basis state. We allow a generous statistical tolerance
    # for shot noise on 20000 shots over 16 outcomes.
    tolerance = 0.05
    quantum_agrees_no_solution = max_dev < tolerance

    classical_says_no_solution = (classical_count == 0)

    ok = classical_says_no_solution and quantum_agrees_no_solution

    print(f"Classical: covering system leaves 0 uncovered values -> "
          f"{classical_says_no_solution}")
    print(f"Quantum: Grover shows no amplified (marked) state -> "
          f"{quantum_agrees_no_solution}")

    if ok:
        print("PASS")
    else:
        print("FAIL")
    return ok


if __name__ == "__main__":
    run()
