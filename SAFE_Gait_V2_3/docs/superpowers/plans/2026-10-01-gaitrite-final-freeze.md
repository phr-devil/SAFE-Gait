# GAITRite final data freeze plan

User specification: no training, no algorithm changes; retain P12_SP_7 only with objective unique timing evidence, otherwise exclude; P1/P7/P9 development-only; freeze manifest, subject CV, support/query rules, data/code/split SHA256.

Execution is inline. No Git repository is present. Existing artifacts remain sealed and unchanged.

Before further P12 inspection, the retention rule is fixed: require shared acquisition timestamps or independently documented hardware trigger/clip origins that map bilateral samples and GAITRite events uniquely and preserve chronological event identities. Pressure periodicity, largest overlap, metadata fit scores, filesystem timestamps and video presentation timestamps without a documented sensor-clock link cannot break a whole-cycle ambiguity. If this evidence is unavailable, exclude the entire physical pass; do not choose or edit an offset.

Fixed split design: numeric subject ID ordering, NumPy PCG64 seed7 permutation, five contiguous outer test blocks; each subject appears in outer test exactly once. Validation is the first3 subjects in a cyclic traversal of subsequent test blocks. Remaining subjects train (12/3/4 for four folds,13/3/3 for one). P1/P7/P9 excluded from all formal roles. Inner training OOF uses5 subject blocks per repeat with existing repeat seeds7,17,27; these are inner partition repeats, not three new outer experiments. Freeze assignments and validate training repeated/paired subject requirements without loading target values into the assignment function.

Preserve frontend, primary targets and CP/simple bias/DS/V2.3 by snapshot and hashes. Natural NP maps to SP task analogue, fast FP maps to HP; slow SP remains a prespecified extension. Do not change methods to fit all three conditions jointly: V2.3 requires two conditions; any extension uses separately declared binary pairing. Current primary analysis remains NP/FP.

Tasks:
1. Audit P12 file/header/timing/event evidence and save exclusion decision.
2. Generate all-record disposition and confirmatory-only manifest, pressure-only input index, condition counts, all9 support/query combination inventories and disjoint physical episodes.
3. Generate outer/inner assignments, feasibility checks, frozen protocol and code snapshots.
4. Independently verify subject isolation, exact coverage, counts, identities, source hashes, prior seals and reproducibility; seal outputs and record bundle hashes. No fitting or inference.
