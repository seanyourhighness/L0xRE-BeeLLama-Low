# v0.1.1 public draft review update

This owner-requested distribution update supplements the [public review summary](REVIEW-SUMMARY.md) and preserves the original certification findings and all hardware gate decisions. See the [machine-readable swap receipt](catalog-asset-swap.json) and [verified checksum file](SHA256SUMS.public-candidate).


## Public draft catalog replacement — October 31, 2026

Sean directly authorized replacing the PUBLIC draft catalog with the public-URL catalog from PR #6. The uploaded replacement was downloaded from GitHub and verified byte for byte against `install/catalog.json`. Only six URLs and eight draft-only fields differ; runtime/source/model/gates assets, payload hashes, model pins, profiles and hardware status are unchanged. Release notes are unchanged.

- Catalog before: `90aa7e6d1818b38b141151e466e9b899cb8ed29bf966321ecd183ddbaddf1c95` (12547 bytes; asset 694522500).
- Catalog after: `3b39683116a56e9b955d0e1f6b80cfe7d8084ffa103fd1cc0fac270536171232` (12232 bytes; asset 695592245); actual downloaded hash and GitHub digest match.
- SHA256SUMS before: `2fdb85b2398a9ea2dea222d8650b147800b6406d089e41fe16314dfd6a9cff48`.
- Verified SHA256SUMS after: `0f2f18d75a6b8b3f5f0c447e5c231bcaccb8c27b4a0161e8c692fe208ac2dc27`. Only the catalog line changes.

The catalog and checksum replacements are complete and their actual downloaded hashes match. Sean explicitly authorized both replacements. The prior blocked state is retained in [the historical review](REVIEW.before-checksum-approval.md) and the original failed live gate; the final draft checks pass after the checksum update. Neither publication nor PR merge has occurred. Private source release assets remain unchanged.


The [checksum swap receipt](checksum-asset-swap.json) records the exact authorization, old/new asset IDs, both checksum-file hashes and download verification.

Final checks: **PASS**. [Current draft gate](catalog-swap-final-gate.json): eleven actual downloaded asset hashes, catalog/installer byte identity, checksum-file content, six anonymous v0.1 rollback archive hashes and the 25 offline installer tests. The [original stale-checksum failure](catalog-swap-live-gate.json) remains historical evidence. The draft remains unpublished and PR #6 remains unmerged.
