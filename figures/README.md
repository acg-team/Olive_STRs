# Visualization scripts

This directory contains figure-generation code, organized in the same numbered
order as the analysis modules under `scripts/`. Visualization programs read
tables produced by the corresponding analysis module and write generated files
to an ignored `output/` directory unless another output path is supplied.

| Directory | Manuscript figures |
|---|---|
| `01_str_landscape` | Figure 2; Figures S1–S2 |
| `02_population_structure` | Figure 3; Figure S3 |
| `03_embeddings` | Figure 4 components; Figures S4–S8 |
| `04_cross_model_comparison` | Cross-model locus-property panels |
| `05_robustness_controls` | Figure S9 |
| `06_trait_association` | Figure 5 components; Figures S10–S13 |
| `07_association_embedding_integration` | Association–embedding panels in Figure 5 |

Some analysis entry points also produce diagnostic plots while calculating
their result tables. Those mixed analysis programs remain under `scripts/`;
the files here are the dedicated visualization entry points.

