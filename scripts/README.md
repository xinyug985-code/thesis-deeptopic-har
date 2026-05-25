````markdown
# DeepTopic-HAR

Deep learning analysis of fetal brain regulatory grammar and Human Accelerated Regions (HARs) using single-cell ATAC-seq topic models.

This repository contains the full analysis pipeline used for my master thesis project, including:

- pycisTopic topic modeling
- CREsted topic-classification models
- motif grammar discovery with TF-MoDISco
- topic-to-celltype mapping
- human vs macaque comparison
- HAR scoring and interpretation
- motif gain/loss analysis
- in silico mutagenesis (ISM)

Main biological goal:

> Identify cell-type-associated regulatory grammar in fetal brain chromatin accessibility and use it to interpret evolutionary changes in Human Accelerated Regions (HARs).

---

# Project Overview

The workflow is organized into two major parts:

## Part 1 — Regulatory Grammar Discovery

Using fetal brain scATAC-seq topics as labels, CREsted CNN models are trained to predict topic accessibility directly from DNA sequence.

Model interpretation is then used to identify:

- topic-associated motifs
- cell-type-associated motif modules
- conserved vs species-biased regulatory grammar

Main frameworks:

- pycisTopic
- CREsted
- TF-MoDISco-lite
- TomTom motif matching

Species:

- Human fetal brain
- Macaque fetal brain

---

## Part 2 — HAR Evolutionary Analysis

Human Accelerated Regions (HARs) are scored using trained sequence models to identify:

- predicted cell-state specificity
- motif gain/loss
- human-specific accessibility shifts
- candidate regulatory mechanisms

Selected HARs are further analyzed using:

- contribution scores
- sequence alignment
- motif annotation
- in silico mutagenesis (ISM)
- humanized/macaquized sequence editing

---

# Repository Structure

```text
scripts/
├── preprocessing/       # Build h5ad and prepare datasets
├── training/            # Train CREsted topic-classification models
├── evaluation/          # Model evaluation and prediction
├── motif_analysis/      # Contribution scores + TF-MoDISco
├── har_analysis/        # HAR scoring and ISM
├── plotting/            # Figure generation
├── exploratory/         # Debugging / experimental scripts
└── useless/             # Old or deprecated scripts
````

---

# Main Pipeline

# 1. Build Topic Datasets

Generate CREsted-compatible topic-by-region AnnData objects.

Main outputs:

* `human_topics.h5ad`
* `macaque_topics.h5ad`

Scripts:

```text
01_make_h5ad_human.py
01_make_h5ad_macaque.py
```

---

# 2. Train Topic-Classification Models

Train CNN models to predict topic accessibility from sequence.

Scripts:

```text
02_train_human.py
02_train_macaque.py
```

Outputs:

```text
final_model.keras
```

Models are trained using:

* 500 bp sequence windows
* CREsted deeptopic CNN
* chromosome-based train/val/test split

---

# 3. Evaluate Models

Evaluate topic prediction performance.

Scripts:

```text
03_eval_human.py
03_eval_macaque.py
```

Typical outputs:

* AUROC
* confusion matrices
* topic prediction summaries

---

# 4. Motif Discovery (Stage2)

This is the core regulatory grammar analysis step.

Scripts:

```text
02_stage2_topic_modisco_top1000.py
```

Main operations:

* generate prediction layers
* build combined score matrix
* region specificity filtering using Gini
* contribution score calculation
* TF-MoDISco motif discovery

Outputs:

* contribution scores
* motif clusters
* pattern matrices
* motif logos

---

# 5. Motif Annotation

Scripts:

```text
03_stage3_summarize_modisco_h5.py
04_stage3_modisco_pattern_matching.py
05_stage4_process_patterns_matrix.py
```

Main outputs:

* motif × topic matrices
* motif × celltype matrices
* TF annotation tables
* clustered motif modules

---

# 6. HAR Analysis

Scripts:

```text
34_2_vs_ism.py
36_2_brain.py
36_5_pick.py
36_8_val.py
```

Main analyses:

* HAR topic scoring
* human vs macaque comparison
* motif gain/loss
* contribution score interpretation
* ISM mutation analysis

---

# Biological Interpretation Strategy

The workflow follows this logic:

```text
sequence
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
HAR evolutionary interpretation
```

Topics are first learned computationally using pycisTopic.

Topics are then mapped back to biological cell states using:

* topic annotation
* topic binarization
* cell-topic enrichment

Therefore:

```text
Model output = regulatory topics
Biological interpretation = cell states / lineages
```

---

# Species Comparison

Human and macaque models are analyzed in parallel to identify:

* conserved motif grammar
* species-biased motif usage
* human-specific regulatory changes

Matched cell states are compared across species rather than directly comparing topic IDs.

---

# Main Dependencies

Core packages:

```text
pycisTopic
CREsted
tensorflow / keras
torch backend
TF-MoDISco-lite
pandas
numpy
scanpy
anndata
matplotlib
seaborn
```

Additional tools:

```text
TomTom (MEME suite)
```

---

# Compute Environment

Main HPC systems:

* Dardel (topic modeling / preprocessing)
* Alvis (GPU model training and interpretation)

Typical GPU:

```text
A100
```

---

# Notes

* Large files are intentionally excluded from GitHub.
* Models, h5ad files, genomes, and intermediate outputs are stored separately.
* Many exploratory plotting/debugging scripts are archived in `exploratory/`.

---

# Thesis Focus

This repository supports the thesis direction:

> Build a fetal brain regulatory grammar atlas using sequence models and apply it to interpret evolutionary changes in Human Accelerated Regions.

Main emphasis:

* motif grammar
* cell-state specificity
* human vs macaque comparison
* HAR mechanistic interpretation

'''
'''

