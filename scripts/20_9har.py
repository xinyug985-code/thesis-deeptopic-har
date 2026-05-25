import pandas as pd

human = pd.read_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/active_HARs_qc_pass.bed", sep="\t", header=None)
macaque = pd.read_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/data/output_rheMac10.bed", sep="\t", header=None)

# index = 3
human_ids = set(human.iloc[:, 3].astype(str))
macaque_ids = set(macaque.iloc[:, 3].astype(str))

lost = human_ids - macaque_ids

print("lost HAR:")
for x in lost:
    print(x)

lost_df = human[human[3].isin(lost)]
lost_df.to_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/lost_hars.csv", index=False, header=False)