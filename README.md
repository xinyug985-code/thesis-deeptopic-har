对，直接在 GitHub 网页里改最方便。
我给你一个更正式、更像论文代码仓库的版本，你直接全覆盖 README.md 就行。

````markdown
# DeepTopic-HAR: Fetal Brain Regulatory Grammar and Human Accelerated Region Analysis

This repository contains the computational pipeline developed for a master thesis project focused on regulatory sequence grammar in fetal brain chromatin accessibility and the functional interpretation of Human Accelerated Regions (HARs).

The project combines:

- single-cell ATAC-seq topic modeling
- deep learning sequence models
- motif grammar discovery
- cross-species comparison
- HAR accessibility prediction
- motif gain/loss interpretation

Main species analyzed:

- Human fetal brain
- Macaque fetal brain

---

# Biological Goal

The main objective of this project is to identify cell-type-associated regulatory grammar in developing brain chromatin accessibility and use it to interpret evolutionary sequence changes in Human Accelerated Regions (HARs).

The overall framework is:

```text
DNA sequence
    ↓
topic accessibility prediction
    ↓
contribution scores
    ↓
motif discovery
    ↓
motif grammar
    ↓
cell-type interpretation
    ↓
HAR evolutionary analysis
````

---

# Workflow Overview

The project consists of two connected components.

## 1. Regulatory Grammar Discovery

Single-cell ATAC-seq data are processed using pycisTopic to generate regulatory topics.

CREsted CNN models are then trained to predict topic accessibility directly from DNA sequence.

Model interpretation methods are used to identify:

* topic-associated motifs
* cell-type-associated motif modules
* conserved and species-biased regulatory grammar

Main tools:

* pycisTopic
* CREsted
* TF-MoDISco-lite
* TomTom (MEME suite)

---

## 2. HAR Evolutionary Analysis

Human Accelerated Regions (HARs) are scored using trained sequence models to identify:

* predicted accessibility shifts
* motif gain/loss
* human-specific regulatory signatures
* candidate developmental regulatory mechanisms

Selected HARs are further analyzed using:

* contribution scores
* motif annotation
* sequence alignment
* in silico mutagenesis (ISM)

---

# Repository Structure

```text
upstream_pycistopic/
    pycisTopic preprocessing and topic-modeling pipeline

scripts/
    downstream CREsted, motif, and HAR analysis scripts

envs/
    conda environment exports for reproducibility

exploratory/
    debugging and exploratory scripts

useless/
    deprecated or obsolete scripts
```

---

# Upstream Pipeline (pycisTopic)

The upstream pipeline generates topic annotations from single-cell ATAC-seq data.

Typical workflow:

```text
step0_prepare_manifest
    ↓
step1_make_cobj
    ↓
step2_run_lda
    ↓
step3_export_beds
    ↓
step4_region_analysis
    ↓
step5_figures
    ↓
step6_topic_annotation
```

Main outputs:

* cisTopic objects
* LDA topic models
* topic BED files
* topic annotation tables

Main environment:

```text
topic
```

Environment files:

```text
envs/topic_env.yml
envs/topic_env_minimal.yml
```

---

# Downstream Pipeline (CREsted)

CREsted models are trained to predict topic accessibility from DNA sequence.

Main downstream analyses include:

* topic prediction
* contribution scores
* TF-MoDISco motif discovery
* motif clustering
* motif-to-celltype mapping
* HAR scoring
* ISM analysis

Main environment:

```text
work
```

Additional motif environments:

```text
motif_env
meme_env
```

---

# Main Analysis Components

## Topic Classification

Train CNN models using topic accessibility labels.

Outputs:

* trained Keras models
* prediction layers
* evaluation metrics

---

## Motif Grammar Discovery

Contribution scores are used for TF-MoDISco motif discovery and motif clustering.

Outputs:

* motif PWMs
* motif clusters
* topic-pattern matrices
* celltype-pattern matrices

---

## HAR Analysis

HARs are scored using trained sequence models to identify candidate human-specific regulatory changes.

Main analyses:

* human vs macaque accessibility prediction
* motif gain/loss
* contribution score interpretation
* in silico mutagenesis (ISM)

---

# Compute Environment

Main HPC systems used:

| System | Purpose                                       |
| ------ | --------------------------------------------- |
| Dardel | pycisTopic preprocessing and topic modeling   |
| Alvis  | CREsted model training and motif/HAR analysis |

Typical GPU:

```text
NVIDIA A100
```

---

# Reproducibility

Conda environment exports are provided in:

```text
envs/
```

Main exported environments:

```text
topic_env.yml
work_env.yml
motif_env.yml
meme_env.yml
```

---

# Notes

* Large intermediate files are intentionally excluded from GitHub.
* Models, large matrices, genomes, and temporary outputs are stored separately on HPC systems.
* The repository focuses on reproducible scripts, environments, and analysis workflows.

---

# Thesis Focus

Main emphasis:

* regulatory grammar
* fetal brain chromatin accessibility
* topic-based sequence modeling
* motif interpretation
* cross-species comparison
* Human Accelerated Regions (HARs)

---

# Author

Xinyu Gao

Master thesis project in computational regulatory genomics.

```
```
