from collections import Counter
from typing import List, Dict
import numpy as np
def extract_acc_norm_values(report_text: str) -> Dict[int, float]:
    """
    Extract acc_norm values for 4 tasks and map them directly to cluster IDs (0 to 3).
    """
    task_to_cluster = {
        "elmb_chatrag": 0,
        "elmb_functioncalling": 1,
        "elmb_reasoning": 2,
        "elmb_roleplay": 3
    }
    acc_norms = {}
    current_task = None
    for line in report_text.strip().splitlines():
        if not line.strip().startswith("|") or "Metric" in line or "---" in line:
            continue
        parts = [p.strip() for p in line.strip().split("|") if p.strip()]
        if not parts:
            continue
        if parts[0].startswith("elmb_"):
            current_task = parts[0].lower()
        if "acc_norm" in parts and current_task in task_to_cluster:
            try:
                value_index = parts.index("acc_norm") + 2
                acc_value = float(parts[value_index])
                cluster_id = task_to_cluster[current_task]
                acc_norms[cluster_id] = acc_value
            except (IndexError, ValueError):
                continue
    return acc_norms
def compute_cluster_distribution(clusters: List[int]) -> List[float]:
    """
    Returns cluster distribution (length 4), divided by 10.
    """
    counts = Counter(clusters)
    return [round(counts.get(i, 0) / 10.0, 2) for i in range(4)]
# === Main Script ===
if __name__ == "__main__":
    report_text = """
    |       Tasks        |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
    |--------------------|------:|------|-----:|--------|---|-----:|---|-----:|
    |elmb_chatrag        |      1|none  |     0|acc     |↑  |0.6375|±  |0.0052|
    |                    |       |none  |     0|acc_norm|↑  |0.7216|±  |0.0049|
    |elmb_functioncalling|      1|none  |     0|acc     |↑  |0.3375|±  |0.0237|
    |                    |       |none  |     0|acc_norm|↑  |0.3925|±  |0.0244|
    |elmb_reasoning      |      1|none  |     0|acc     |↑  |0.3201|±  |0.0208|
    |                    |       |none  |     0|acc_norm|↑  |0.2883|±  |0.0202|
    |elmb_roleplay       |      1|none  |     0|acc     |↑  |0.7149|±  |0.0052|
    |                    |       |none  |     0|acc_norm|↑  |0.5883|±  |0.0057|
    """
    cluster_assignments = [653, 613, 174, 447, 1667, 1420, 264, 2639, 1028, 521, 932, 758, 1140, 487, 32, 1599, 1096, 815, 500, 1304]
    cluster_assignments=np.array(cluster_assignments)
    acc_norms = extract_acc_norm_values(report_text)
    acc_norm_array = [acc_norms[i] for i in range(4)]
    # :white_check_mark: Final Output
    print("acc_norm values:")
    print(acc_norm_array)
    print("\nCluster distribution (divided by 10):")
    print(cluster_assignments/1000)

