# v0.1.1 public draft review update

This owner-requested distribution update supplements the [public review summary](REVIEW-SUMMARY.md) and preserves the original certification findings and all hardware gate decisions. See the [machine-readable swap receipt](catalog-asset-swap.json) and [prepared checksum file](SHA256SUMS.public-candidate).


## Public draft catalog replacement — October 31, 2026

Sean directly authorized replacing the PUBLIC draft catalog with the public-URL catalog from PR #6. The uploaded replacement was downloaded from GitHub and verified byte for byte against `install/catalog.json`. Only six URLs and eight draft-only fields differ; runtime/source/model/gates assets, payload hashes, model pins, profiles and hardware status are unchanged. Release notes are unchanged.

- Catalog before: `90aa7e6d1818b38b141151e466e9b899cb8ed29bf966321ecd183ddbaddf1c95` (12547 bytes; asset 694522500).
- Catalog after: `3b39683116a56e9b955d0e1f6b80cfe7d8084ffa103fd1cc0fac270536171232` (12232 bytes; asset 695592245); actual downloaded hash and GitHub digest match.
- SHA256SUMS before: `2fdb85b2398a9ea2dea222d8650b147800b6406d089e41fe16314dfd6a9cff48`.
- Prepared SHA256SUMS after: `0f2f18d75a6b8b3f5f0c447e5c231bcaccb8c27b4a0161e8c692fe208ac2dc27`. Only the catalog line changes.

The catalog swap is complete. The public draft is NOT ready to publish while its uploaded SHA256SUMS still lists the old catalog hash. The corrected checksum file is prepared, but replacing that additional asset requires Sean's exact-list authorization under his standing release-asset deletion rule. No publication or PR merge has occurred. The private source release assets are unchanged.
