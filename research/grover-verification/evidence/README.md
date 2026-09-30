# Historical simulator evidence

The two directories contain unchanged per-arm outputs and circuit exports from
the September 20, 2026 local v2 study. Both used the same seeds. Matching counts
demonstrate replay, not independent statistical replication. The known-target
positive arm recorded 3929/4096 target-13 outcomes; the wrong-oracle arm recorded
4/4096 and the uniform classical reference 267/4096.

These are selected historical artifacts, not complete portable run receipts.
Original frozen manifests and operational receipts remain private because they
contain machine-specific paths. Their omission is deliberate; do not describe
this subset as the original complete reproduction package or T9 acceptance.

The published study changes the historical source default to a repository-relative
path and removes a machine-specific README command. Source hashes therefore
differ from the historical study. Prepare and independently review a new manifest
before executing this public version. Do not modify old manifests to make them
accept changed source. No new simulator execution is claimed by this upload.

The uniform classical arm is not an optimized classical competitor. No hardware,
quantum advantage, unknown-answer discovery, or number-theory proof is established.
