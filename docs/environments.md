# Software environments

`environments/analysis.yml` supports the CPU population, landscape,
association, and plotting analyses. The mixed-model implementation reported in
the manuscript used Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, and SciPy 1.15.2.

The three genomic foundation models are isolated because their upstream
packages and GPU requirements differ:

- `agront.yml`: AgroNT inference and regression;
- `plantcad2.yml`: PlantCAD2-Large inference and regression;
- `gpn.yml`: GPN inference and regression.

Model repository names and immutable revisions are in
`config/project.example.yaml`. Hugging Face cache locations are local settings
and must not be committed. CUDA builds may need adjustment for the host driver;
changing the CUDA build does not change the pinned model revision.

ADMIXTURE, PLINK 1.9, BEDTools, and samtools are external command-line tools.
Their executable paths may be supplied directly to the relevant scripts.

