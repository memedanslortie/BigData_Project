import os
import json
import plotly.graph_objects as go

def plot_all_methods(result_dir="results/"):
    all_results = []

    for root, _, files in os.walk(result_dir):
        for file in files:
            if file.endswith(".json"):
                with open(os.path.join(root, file)) as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            all_results.extend(content)
                    except Exception as e:
                        print(f"Erreur dans {file} : {e}")

    method_groups = {}
    for res in all_results:
        method = res["algorithm"]
        method_groups.setdefault(method, []).append(res)

    fig = go.Figure()


    for method, res_list in method_groups.items():
        recall_values = []
        qps_values = []
        hover_texts = []

        for res in sorted(res_list, key=lambda x: x["metrics"]["recall@10"]):
            recall = res["metrics"]["recall@10"]
            qps = 1 / res["metrics"]["search_time"]

            param_str = ", ".join(f"{k}: {v}" for k, v in res["parameters"].items())

            hover_text = (
                f"<b>{method}</b><br>"
                f"Recall@10: {recall:.4f}<br>"
                f"QPS: {qps:.2f}<br>"
                f"{param_str}"
            )

            recall_values.append(recall)
            qps_values.append(qps)
            hover_texts.append(hover_text)

        fig.add_trace(go.Scatter(
            x=recall_values,
            y=qps_values,
            mode='lines+markers',
            name=method,
            hovertext=hover_texts,
            hoverinfo='text',
            line_shape='spline'
        ))

    fig.update_layout(
        title="Recall vs Queries per second",
        xaxis_title="Recall@10",
        yaxis_title="Queries per second (log scale)",
        yaxis_type="log",
        xaxis=dict(range=[0.0, 1.0]),
        hovermode="closest"
    )

    fig.show()

plot_all_methods()
