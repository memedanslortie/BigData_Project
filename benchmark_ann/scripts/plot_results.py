import json
import plotly.graph_objects as go

def plot_interactive_faiss(filename="results/benchmark_faiss_results.json"):
    with open(filename, "r") as f:
        results = json.load(f)

    method_groups = {}
    for res in results:
        method = res["method"]
        if method not in method_groups:
            method_groups[method] = []
        method_groups[method].append(res)

    fig = go.Figure()

    for method, res_list in method_groups.items():
        recall_values = []
        qps_values = []
        hover_texts = []

        for res in sorted(res_list, key=lambda x: x["recall"]):
            recall = res["recall"]
            
            qps = 1 / res["search_time"]
            nlist = res["faiss_params"]["nlist"]
            nprobe = res["faiss_params"]["nprobe"]
    
            hover_text = (
                f"<b>{method}</b><br>"
                f"Recall@{res['k']}: {recall:.4f}<br>"
                f"QPS: {qps:.2f}<br>"
                f"nlist: {nlist}, nprobe: {nprobe}"
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
            hoverinfo='text'
        ))

    fig.update_layout(
        title="Recall vs Queries per Second",
        xaxis_title="Recall@k",
        yaxis_title="Queries per second (log scale)",
        yaxis_type="log",
        xaxis=dict(range=[0.0, 1.0]),  
        hovermode="closest"
    )

    fig.show()

plot_interactive_faiss()
