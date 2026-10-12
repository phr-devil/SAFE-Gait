# GAITRite external manifest audit plan

Goal/spec: current user request; inspect local insole-gaitrite-dataset, build a traceable external manifest, assess conditions/repeats/reference and formal validation readiness. No training or inference, no edits to frozen four methods, no use of WearGait test outcomes for mapping or quality rules.

Architecture: read raw CSV/JSON and official acquisition documentation; preserve source subject/condition/clip identity. Write an audit-only manifest, separate structural availability from synchronization/reference verification and frozen-model compatibility. No raw channel concatenation or conversion into WearGait inputs.

Execution: native in this session, using executing-plans and verification-before-completion; final independent review per requesting-code-review. No Git repository; preserve source hashes.

- [x] Inspect local hierarchy, archive, existing adapter, official protocol and synchronization semantics. Map NP natural to WearGait self-paced SP; FP fast to HP as a task analogue, not identical protocol. Keep dataset SP slow distinct.
- [x] Audit every walking clip: bilateral32 channels, numeric/finite data, sampling provenance, primary reference fields, reference IDs/summary joins, synchronization coverage and duplicate physical pass identities. Save external_manifest.csv, subject/condition/pair summaries, source hashes, issues and audit report.
- [x] Independently verify counts, uniqueness, support/query disjointness and repeated-pass eligibility. Distinguish data-eligible counts from ready-for-model counts; document unverified source facts instead of guessing.
- [x] Freeze the audit artifacts and report only requested counts, combinations, reference reliability, readiness and next steps. No training, no gate, no target whitelist change.

Review focus: literal SP name collision;32perfoot vs32total;quoted multiline GAITRite CSV;condition-summary versus individual-pass references;missing/nonfinite channels;duplicate or mismatched pass IDs;sync metadata not proof of accurate alignment;no repeated-pass variance fitting on external evaluation subjects.

Completed evidence:22subjects664walkingpasses,SP228/NP219/FP217;P16FPabsent. 6tests pass;independent verification checks14,228,992pressurecells,664reference geometry/timing sets,1907disjoint candidateepisodes,2723sourcehashes and1430unchanged priorfrozenartifacts. Independent reviewer closed P7_FP_1 truncated-leading-event report issue; P7FP36summary/34storedpositive/33bounded steps is now explicit. No remaining material findings. Formal readiness false due to incompatible64channel device input and unverified synchronization/target-definition prerequisites. No training/inference performed.
