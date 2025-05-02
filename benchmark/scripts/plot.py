import os
import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_all_datasets(base_dir="/Volumes/SSD/M1VMI/S2/big_data/results"):
    """
    Génère une visualisation des résultats de benchmark pour tous les datasets disponibles,
    chacun dans son propre sous-graphique, empilés verticalement.
    """
    datasets = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and 
                any(f.endswith('.json') for f in os.listdir(os.path.join(base_dir, d)))]
    
    if not datasets:
        print(f"Aucun dataset trouvé dans {base_dir}")
        return

    n_datasets = len(datasets)
    
    fig = make_subplots(
        rows=n_datasets, 
        cols=1,
        subplot_titles=datasets,
        vertical_spacing=0.12
    )
    
    all_methods = set()
    method_colors = {}
    color_sequence = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    # Traiter chaque dataset
    for idx, dataset in enumerate(datasets):
        row = idx + 1
        col = 1
        
        result_dir = os.path.join(base_dir, dataset)
        all_results = []
        
        # Chargement des résultats depuis les fichiers JSON
        for file in os.listdir(result_dir):
            if file.endswith(".json"):
                with open(os.path.join(result_dir, file)) as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            all_results.extend(content)
                    except Exception as e:
                        print(f"Erreur dans {dataset}/{file} : {e}")
        
        if not all_results:
            print(f"Aucun résultat pour {dataset}")
            continue
            
        # Regroupement par algorithme
        method_groups = {}
        for res in all_results:
            method = res["algorithm"]
            all_methods.add(method)
            method_groups.setdefault(method, []).append(res)
        
        # Assigner des couleurs cohérentes aux méthodes
        for i, method in enumerate(all_methods):
            if method not in method_colors:
                method_colors[method] = color_sequence[i % len(color_sequence)]
        
        # Traitement de chaque algorithme
        for method, res_list in method_groups.items():
            recall_values = []
            qps_values = []
            hover_texts = []
            
            # Extraction des données
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
            
            # Calculer la frontière de Pareto
            if len(recall_values) >= 2:
                # Identifier les points Pareto-optimaux (non dominés)
                pareto_indices = []
                for i in range(len(recall_values)):
                    is_dominated = False
                    for j in range(len(recall_values)):
                        if i != j:
                            if ((recall_values[j] >= recall_values[i] and qps_values[j] > qps_values[i]) or
                                (recall_values[j] > recall_values[i] and qps_values[j] >= qps_values[i])):
                                is_dominated = True
                                break
                    if not is_dominated:
                        pareto_indices.append(i)
            
                pareto_indices.sort(key=lambda i: recall_values[i])
                
                pareto_recalls = [recall_values[i] for i in pareto_indices]
                pareto_qps = [qps_values[i] for i in pareto_indices]
                pareto_texts = [hover_texts[i] for i in pareto_indices]
                
                show_legend = (idx == 0)
                
                # Tracer la frontière de Pareto
                fig.add_trace(
                    go.Scatter(
                        x=pareto_recalls,
                        y=pareto_qps,
                        mode='lines+markers',
                        name=method,
                        hovertext=pareto_texts,
                        hoverinfo='text',
                        line=dict(width=2, color=method_colors[method]),
                        marker=dict(size=8, color=method_colors[method]),
                        showlegend=show_legend
                    ),
                    row=row, col=col
                )
                
                # Ajouter tous les points en arrière-plan avec transparence
                fig.add_trace(
                    go.Scatter(
                        x=recall_values,
                        y=qps_values,
                        mode='markers',
                        marker=dict(size=4, color='rgba(150,150,150,0.3)'),
                        name=method + " (all)",
                        hovertext=hover_texts,
                        hoverinfo='text',
                        showlegend=False
                    ),
                    row=row, col=col
                )
            else:
                show_legend = (idx == 0)
                fig.add_trace(
                    go.Scatter(
                        x=recall_values,
                        y=qps_values,
                        mode='lines+markers',
                        name=method,
                        hovertext=hover_texts,
                        hoverinfo='text',
                        line_shape='spline',
                        line=dict(color=method_colors[method]),
                        showlegend=show_legend
                    ),
                    row=row, col=col
                )
        
        fig.update_xaxes(
            title_text="Recall@10" if row == n_datasets else "",
            range=[0.0, 1.0],
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="QPS (log)",
            type="log",
            row=row, col=col
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=row, col=col)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)', row=row, col=col)
    
    fig.update_layout(
        title_text="Comparaison des Algorithmes ANN sur Différents Datasets - Frontières de Pareto",
        height=400 * n_datasets + 100, 
        width=1000,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        hovermode="closest"
    )
    
    fig.show()
    
    html_file = "all_datasets_pareto_comparison.html"
    fig.write_html(html_file)
    print(f"Graphique sauvegardé dans {html_file}")

if __name__ == "__main__":
    plot_all_datasets()