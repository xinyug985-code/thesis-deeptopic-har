import anndata as ad
from pathlib import Path

H5AD = Path("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/human_topics.h5ad")
adata = ad.read_h5ad(H5AD)
print(adata.shape)
print(adata.obs_names[:10])
print(adata.var_names[:10])