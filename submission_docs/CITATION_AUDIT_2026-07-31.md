# Citation audit

Audit date: 2026-07-31.

## Result

- All 22 journal, conference, and book-chapter records in `references.bib`
  returned exact-title matches from Crossref using their stored DOI.
- Two records were also confirmed by Semantic Scholar during the automated
  multi-source run. The remaining Crossref-only labels are coverage/rate-limit
  warnings, not title or DOI mismatches.
- The BigMEC paper was directly confirmed as `BigMEC: Scalable Service
  Migration for Mobile Edge Computing`, DOI
  `10.1109/SEC54971.2022.00018`.
- The BigMEC software archive was directly confirmed through DataCite as
  `flbrandh/MEC-Simulator-2-BigMEC: v1.0`, DOI
  `10.5281/zenodo.10810301`, publisher Zenodo, publication year 2024.
- The software archive's Zenodo creator is `flbrandh`; it is not the same
  author list as the BigMEC conference paper. The BibTeX record therefore
  cites the archive under its recorded software creator and must not be read
  as a four-author software publication.

## Tool boundary

The general citation checker queries Crossref, Semantic Scholar, and OpenAlex,
but not DataCite. It therefore marked the Zenodo software record as not found.
That software warning is resolved by the direct DataCite record above. No DOI
or title mismatch remains unresolved.
