# Manuscript mapping

| Manuscript item | Code |
|---|---|
| STR annotation and genome-wide landscape | `scripts/01_str_landscape/` |
| Figure 2; Figures S1–S2 | `figures/01_str_landscape/` plus the landscape analysis tables |
| SNP/STR population structure | `scripts/02_population_structure/` |
| Figure 3; Figure S3 | `figures/02_population_structure/` using PCA, ADMIXTURE, dNMF, component-alignment, and geography outputs |
| Reference-anchored allele embeddings | `scripts/03_embeddings/` |
| Figures S4–S8 | Model-specific tables under `scripts/03_embeddings/`; visualization under `figures/03_embeddings/` |
| Figure 4 | `scripts/04_cross_model_comparison/` and `figures/04_cross_model_comparison/` |
| Fixed-length and matched-indel controls; Figure S9 | `scripts/05_robustness_controls/` and `figures/05_robustness_controls/` |
| STR–oil-quality association; Figures S10–S12 | `scripts/06_trait_association/` and `figures/06_trait_association/` |
| Figure 5; Figure S13 | `figures/06_trait_association/` and `figures/07_association_embedding_integration/` |

Figure assembly may combine panels produced by more than one script. The
script-level manifest records the principal input and output contract for each
stage.

