# Verification snapshots

These records document distinct stages of the same study. Historical job counts
describe their recorded stage, rather than the current completed state.

- `diagnosis.json`: batch-composition diagnosis of the original fixed-refit
  failure, before any executor changes.
- `completed_jobs_verification.json`: original 152 completed jobs (150 neural
  and two fixed), with 762 verified artifact checksums and recomputed neural
  metrics before recovery.
- `repair_verification.json`: 68 passing automated tests and isolated fold-3
  GPU checks for all four fixed finalists. Shortened diagnostic benchmarks were
  not used as study results.
- `recovery_preparation_verification.json`: all 150 neural jobs reused with
  matching identities, unchanged original metadata/manifests and linked model
  artifacts; the 40 fixed benchmarks were still pending at that stage.
- `completed_study_verification.json`: final completed-job, artifact, canonical
  metric and analysis-input checks after the author's full execution.

The first four JSON files are byte-preserving copies of reports originally
written under `cache/fixed-refit-diagnostics/`. Paths inside those snapshots
refer to that original diagnostic location. Final study and analysis artifacts
are in the parent directory and `../manuscript_tables/`.

Large model/checkpoint/pipeline binaries are kept on the training workstation.
The corresponding job manifests retain their hashes; the Git report archive
supports statistical reanalysis but requires those binaries for deployment or
training recovery.
