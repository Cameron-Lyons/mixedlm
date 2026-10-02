# lme4 dataset provenance

These compressed CSVs contain the original tables in the `data/*.rda` files of
[lme4 commit 67d71b0e264bda95f22bdc3ec52261c5fc993d4a](https://github.com/lme4/lme4/tree/67d71b0e264bda95f22bdc3ec52261c5fc993d4a/data).
The upstream package reports version `2.1-0` and license `GPL (>= 2)` in its
[DESCRIPTION](https://github.com/lme4/lme4/blob/67d71b0e264bda95f22bdc3ec52261c5fc993d4a/DESCRIPTION).
The GPL version 2 text is included in `LICENSE`; this dataset license is separate
from the mixedlm core package's MIT license.
Source attribution and study references are available in the corresponding
[lme4 dataset documentation](https://lme4.github.io/lme4/reference/index.html).

`provenance.json` records the immutable source URL and SHA256 of every RDA file,
the compressed and uncompressed CSV SHA256, the original schema, and row count.
Tables were decoded with pyreadr 0.5.7 and written with pandas `to_csv(index=False,
float_format="%.17g", lineterminator="\n")`, then compressed with gzip level 9
and `mtime=0`. Numeric CSV loading uses round-trip float parsing. R factor labels
are exposed as ordinary Python strings rather than R factor codes or ordered
pandas categories. Original row and column order are retained.

The CSV assets contain only original columns. The loaders append two documented
compatibility aliases: Arabidopsis `total_fruits` copies `total.fruits`, and
grouseticks `cTICKS` copies `TICKS`. No network connection, R installation, or
pyreadr dependency is needed to load the bundled data.
