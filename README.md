# Olive STR variation and genomic foundation models

Analysis code for **“Interpreting Short Tandem Repeat Variations with Plant
Genomic Foundation Models in Olive.”** The study characterizes 117,119
polymorphic STRs in 212 olive accessions, compares SNP- and STR-derived
population structure, measures allele-dependent embedding perturbation with
AgroNT, PlantCAD2, and GPN, and integrates those results with oil-composition
associations.

This repository is a publication code archive. STR genotyping pipeline is maintained separately in
[olive_str_genotyping](https://github.com/Garfieldfei/olive_str_genotyping).

## Repository map

| Directory | Contents |
|---|---|
| `scripts/01_str_landscape` | Functional annotation, chromosome density, motif composition, genomic context, and reference-relative variation |
| `scripts/02_population_structure` | SNP/STR PCA, ADMIXTURE, dNMF, ancestry-component concordance, and fine-scale geography |
| `scripts/03_embeddings` | Allele catalogue preparation and AgroNT, PlantCAD2, and GPN embedding analyses |
| `scripts/04_cross_model_comparison` | Matched-locus comparison across the three foundation models |
| `scripts/05_robustness_controls` | Fixed-total-length and matched nonrepetitive-indel controls |
| `scripts/06_trait_association` | STR mixed-model association with olive-oil traits |
| `scripts/07_association_embedding_integration` | Integration of exploratory associations with embedding evidence |
| `figures/` | Dedicated manuscript visualization scripts, grouped by the same numbered modules |

See [`docs/manuscript_mapping.md`](docs/manuscript_mapping.md) for the mapping
to manuscript figures and [`docs/analysis_manifest.tsv`](docs/analysis_manifest.tsv)
for script-level inputs and outputs.

## Data

The associated Zenodo record will contain the
per-sample QC-annotated STR VCFs (will be available soon).

The OLEA9 reference assembly and annotation, public sequencing accessions, and
required table schemas are described in [`docs/input_data.md`](docs/input_data.md).

## Environments

Create the CPU analysis environment with:

```bash
conda env create -f environments/analysis.yml
```

Separate GPU environments are provided for each foundation model. Model
identifiers and immutable revisions are recorded in
[`config/project.example.yaml`](config/project.example.yaml). See
[`docs/environments.md`](docs/environments.md) and
[`docs/hpc_execution.md`](docs/hpc_execution.md) before running GPU jobs.

## Running an analysis

Scripts expose `--help` and accept explicit input/output paths. Copy the public
configuration template before use:

```bash
cp config/project.example.yaml config/project.yaml
python figures/01_str_landscape/plot_panel_a.py --help
python scripts/06_trait_association/analyze_str_oil_association.py --help
```

The model-specific embedding programs retain JSON configuration files because
they record architecture-specific constants. Example JSON files are stored
beside those programs.


