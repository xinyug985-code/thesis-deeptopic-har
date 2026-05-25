import pandas as pd

df = pd.read_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_global/har_with_category.csv")

human_top = (
    df[df["category"] == "human_high"]
    .sort_values("top_score__human_model", ascending=False)
    .head(10)
)

macaque_top = (
    df[df["category"] == "macaque_high"]
    .sort_values("top_score__macaque_model", ascending=False)
    .head(5)
)

conserved = df[df["category"] == "conserved"]

human_specific = (
    df[df["category"] == "human_high"]
    .sort_values("delta_top_score_human_minus_macaque_model", ascending=False)
    .head(10)
)

human_top.to_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_select/human_high.csv", index=False)
macaque_top.to_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_select/macaque_high.csv", index=False)
conserved.to_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_select/conserved.csv", index=False)
human_specific.to_csv("/mimer/NOBACKUP/groups/naiss2025-22-612/xinyu/pjt/runs/out/har_select/human_specific.csv", index=False)