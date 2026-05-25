import pandas as pd

df = pd.read_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_integrated/master_har_3species_2models_wide.csv")

lost_ids = [
    "HARsv2_0021","HARsv2_0327","HARsv2_1082","HARsv2_1136",
    "HARsv2_1309","HARsv2_1586","HARsv2_1690","HARsv2_1943","HARsv2_2153"
]

sub = df[df["har_id"].isin(lost_ids)]

print(sub[[
    "har_id",
    "top_score__human_model",
    "best_celltype_by_mean__human_model",
    "top_score__macaque_model",
    "best_celltype_by_mean__macaque_model"
]])