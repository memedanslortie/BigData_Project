import os
import json
import plotly.graph_objects as go

def plot_recall_vs_time(result_dir="results/"):
    all_results = []

    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path) as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            all_results.extend(content)
                    except Exception as e:
                        print(f"[!] Erreur dans {file} : {e}")

    method_groups = {}
    for res in all_results:
        method = res["algorithm"]
        method_groups.setdefault(method, []).append(res)

    fig = go.Figure()

    for method, res_list in method_groups.items():
        recall_values = []
        time_values = []
        hover_texts = []

        # Trier par recall croissant
        for res in sorted(res_list, key=lambda x: x["metrics"]["recall@10"]):
            recall = res["metrics"]["recall@10"]
            time = res["metrics"]["search_time"] * 1000  # en millisecondes

            params = ", ".join(f"{k}: {v}" for k, v in res["parameters"].items())
            hover_text = (
                f"<b>{method}</b><br>"
                f"Recall@10: {recall:.4f}<br>"
                f"Search time: {time:.3f} ms<br>"
                f"{params}"
            )

            recall_values.append(recall)
            time_values.append(time)
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=recall_values,
            y=time_values,
            mode='lines+markers',
            name=method,
            hovertext=hover_texts,
            hoverinfo='text',
            line_shape='spline'
        ))

    fig.update_layout(
        title="Recall vs Search Time",
        xaxis_title="Recall@10",
        yaxis_title="Search time (ms, log scale)",
        yaxis_type="log",
        xaxis=dict(range=[0.0, 1.0]),
        hovermode="closest"
    )

    fig.show()

plot_recall_vs_time()
