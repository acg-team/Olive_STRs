# Input data

The repository does not redistribute controlled, bulky, or third-party inputs.
Paths are supplied explicitly to scripts or recorded in a copied
`config/project.yaml`.

## Public study data

The associated Zenodo record (`ZENODO_DOI_PENDING`) contains:

- Supplementary Data S1: manifest for the 212 retained accessions;
- Supplementary Data S2: oil-composition phenotypes for 140 accessions;
- Supplementary Data S3: 117,119-locus polymorphic STR catalogue;
- Supplementary Data S4: PCA and selected ADMIXTURE/dNMF coefficients;
- Supplementary Data S5: matched primary three-model embedding results;
- Supplementary Data S6: exploratory STR associations at within-trait FDR < 0.05;
- Supplementary Data S7: embedding robustness summaries; and
- one QC-annotated STR VCF per sample.

The complete embeddings are not required to inspect the published statistical
results. Rerunning model inference requires the prepared allele catalogue and
the model weights obtained from Hugging Face.

## External inputs

- OLEA9 reference FASTA and Ensembl Plants release 60 GFF3.
- Public accessions listed in Supplementary Data S1.
- The chromosome SNP PLINK dataset used for PCA, kinship, and ADMIXTURE.
- PASS-only ConSTRain call tables when rebuilding the embedding allele catalogue.

STR genotyping is performed by the separate
[`olive_str_genotyping`](https://github.com/Garfieldfei/olive_str_genotyping)
workflow. This analysis repository begins from its QC-annotated calls and
summary matrices.

## Identifiers and coordinates

STR identifiers use `contig_start`, where `start` is the zero-based BED start.
Scripts that display loci may render the same value as `contig:start`. Do not
silently convert these starts to one-based coordinates. Sample identifiers must
match Supplementary Data S1 exactly.

