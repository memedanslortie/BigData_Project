import os
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_all_methods(result_dir="results/", dataset=None, algorithms=None, show_pareto=True):
    all_results = []
    
    if dataset:
        dataset_path = os.path.join(result_dir, dataset)
        if os.path.exists(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith(".json"):
                    with open(os.path.join(dataset_path, file)) as f:
                        try:
                            content = json.load(f)
                            if isinstance(content, list):
                                all_results.extend(content)
                        except Exception as e:
                            print(f"Erreur dans {file} : {e}")
    else:
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

    if algorithms:
        all_results = [res for res in all_results if res.get("algorithm") in algorithms]
    
    if not all_results:
        print("Aucun résultat trouvé!")
        return
    
    method_groups = {}
    for res in all_results:
        method = res["algorithm"]
        method_groups.setdefault(method, []).append(res)
    
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    for i, (method, res_list) in enumerate(method_groups.items()):
        recall_values = []
        qps_values = []
        hover_texts = []
        
        for res in sorted(res_list, key=lambda x: x["metrics"].get("recall@10", 0)):
            recall = res["metrics"].get("recall@10", 0)
            search_time = res["metrics"].get("search_time", float('inf'))
            qps = 1 / search_time if search_time > 0 else 0
            
            param_str = ", ".join(f"{k}: {v}" for k, v in res["parameters"].items())

            index_time = res["metrics"].get("index_time", "N/A")
            if index_time != "N/A":
                time_info = f"Index Time: {index_time:.2f}s"
            else:
                time_info = "Index Time: N/A"
            
            hover_text = (
                f"<b>{method}</b><br>"
                f"Recall@10: {recall:.4f}<br>"
                f"QPS: {qps:.2f}<br>"
                f"{time_info}<br>"
                f"<i>Paramètres:</i><br>{param_str}"
            )
            
            recall_values.append(recall)
            qps_values.append(qps)
            hover_texts.append(hover_text)
        
        fig.add_trace(
            go.Scatter(
                x=recall_values,
                y=qps_values,
                mode='lines+markers',
                name=method,
                hovertext=hover_texts,
                hoverinfo='text',
                marker=dict(
                    size=10,
                    color=colors[i % len(colors)],
                    line=dict(width=1, color='black')
                ),
                line=dict(
                    shape='spline',
                    smoothing=0.3,
                    color=colors[i % len(colors)]
                )
            )
        )
    
    if show_pareto and len(method_groups) > 1:
        all_x = []
        all_y = []
        for method, res_list in method_groups.items():
            for res in res_list:
                recall = res["metrics"].get("recall@10", 0)
                search_time = res["metrics"].get("search_time", float('inf'))
                qps = 1 / search_time if search_time > 0 else 0
                all_x.append(recall)
                all_y.append(qps)
        
        points = list(zip(all_x, all_y))
        points.sort()  
        
        pareto_points = []
        max_qps = 0
        for recall, qps in points:
            if qps > max_qps:
                pareto_points.append((recall, qps))
                max_qps = qps
        
        if pareto_points:
            pareto_x, pareto_y = zip(*pareto_points)
            fig.add_trace(
                go.Scatter(
                    x=pareto_x,
                    y=pareto_y,
                    mode='lines',
                    name='Frontière de Pareto',
                    line=dict(color='black', width=2, dash='dash'),
                    hoverinfo='skip'
                )
            )
    
    dataset_title = f" sur {dataset}" if dataset else ""
    fig.update_layout(
        title=f"Recall vs Queries per second{dataset_title}",
        xaxis_title="Recall@10",
        yaxis_title="Queries per second (log scale)",
        yaxis_type="log",
        xaxis=dict(range=[0.0, 1.02]),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",
        margin=dict(l=80, r=80, t=100, b=80),
        plot_bgcolor='white',
    )
    
    stats_text = []
    
    for method, res_list in method_groups.items():
        recalls = [res["metrics"].get("recall@10", 0) for res in res_list]
        qps_values = [1 / res["metrics"].get("search_time", float('inf')) 
                      if res["metrics"].get("search_time", 0) > 0 else 0 
                      for res in res_list]
        
        if recalls and qps_values:
            best_recall = max(recalls)
            best_qps = max(qps_values)
            
            stats_text.append(f"{method}: max recall={best_recall:.4f}, max QPS={best_qps:.0f}")
    
    if stats_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=1.08,
            text="<br>".join(stats_text),
            showarrow=False,
            font=dict(size=10),
            align="left",
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1,
            borderpad=4
        )
    
    fig.show()


def list_available_datasets(result_dir="results/"):
    if not os.path.exists(result_dir):
        print(f"Le répertoire {result_dir} n'existe pas.")
        return []
    
    datasets = [d for d in os.listdir(result_dir) 
                if os.path.isdir(os.path.join(result_dir, d))]
    
    if datasets:
        print("Datasets disponibles:")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset}")
    else:
        print("Aucun dataset trouvé.")
        
    return datasets

def list_available_algorithms(dataset, result_dir="results/"):
    dataset_path = os.path.join(result_dir, dataset)
    
    if not os.path.exists(dataset_path):
        print(f"Le dataset {dataset} n'existe pas.")
        return []
    
    algorithms = set()
    
    for file in os.listdir(dataset_path):
        if file.endswith(".json"):
            with open(os.path.join(dataset_path, file)) as f:
                try:
                    content = json.load(f)
                    if isinstance(content, list):
                        for result in content:
                            if "algorithm" in result:
                                algorithms.add(result["algorithm"])
                except Exception as e:
                    print(f"Erreur dans {file} : {e}")
    
    if algorithms:
        print(f"Algorithmes disponibles pour {dataset}:")
        for i, algo in enumerate(sorted(algorithms), 1):
            print(f"{i}. {algo}")
    else:
        print(f"Aucun algorithme trouvé pour {dataset}.")
        
    return sorted(algorithms)

def interactive_visualization():
    datasets = list_available_datasets()
    
    if not datasets:
        print("Aucun dataset disponible.")
        return
    
    dataset_choice = input("\nSélectionnez un dataset (numéro ou nom, laissez vide pour le premier): ")
    
    if not dataset_choice:
        dataset = datasets[0]
    else:
        try:
            idx = int(dataset_choice) - 1
            if 0 <= idx < len(datasets):
                dataset = datasets[idx]
            else:
                print(f"Numéro invalide. Utilisation du premier dataset: {datasets[0]}")
                dataset = datasets[0]
        except ValueError:
            if dataset_choice in datasets:
                dataset = dataset_choice
            else:
                print(f"Nom de dataset invalide. Utilisation du premier dataset: {datasets[0]}")
                dataset = datasets[0]
    
    algorithms = list_available_algorithms(dataset)
    
    if not algorithms:
        print(f"Aucun algorithme disponible pour {dataset}.")
        return

    print("\nOptions de visualisation:")
    print("1. Tous les algorithmes")
    print("2. Sélectionner un algorithme spécifique")
    print("3. Sélectionner plusieurs algorithmes")
    
    viz_choice = input("\nChoisissez une option (1-3): ")
    
    selected_algos = None
    
    if viz_choice == "1":
        selected_algos = algorithms
    elif viz_choice == "2":
        algo_choice = input(f"\nSélectionnez un algorithme (1-{len(algorithms)}): ")
        try:
            idx = int(algo_choice) - 1
            if 0 <= idx < len(algorithms):
                selected_algos = [algorithms[idx]]
            else:
                print("Numéro invalide. Visualisation de tous les algorithmes.")
                selected_algos = algorithms
        except ValueError:
            if algo_choice in algorithms:
                selected_algos = [algo_choice]
            else:
                print("Nom d'algorithme invalide. Visualisation de tous les algorithmes.")
                selected_algos = algorithms
    elif viz_choice == "3":
        print("\nEntrez les numéros des algorithmes séparés par des espaces")
        print("Exemple: 1 3 pour sélectionner le premier et le troisième algorithme")
        multi_choice = input(f"Sélection (1-{len(algorithms)}): ")
        
        indices = []
        try:
            indices = [int(i) - 1 for i in multi_choice.split()]
            selected_algos = [algorithms[i] for i in indices if 0 <= i < len(algorithms)]
            
            if not selected_algos:
                print("Sélection invalide. Visualisation de tous les algorithmes.")
                selected_algos = algorithms
        except ValueError:
            print("Entrée invalide. Visualisation de tous les algorithmes.")
            selected_algos = algorithms
    else:
        print("Option invalide. Visualisation de tous les algorithmes.")
        selected_algos = algorithms
    
    show_pareto = True
    if len(selected_algos) > 1:
        pareto_choice = input("\nAfficher la frontière de Pareto? (o/n, défaut: o): ").lower()
        show_pareto = pareto_choice != "n"
    
    print(f"\nVisualisation de {', '.join(selected_algos)} sur {dataset}...")
    plot_all_methods(dataset=dataset, algorithms=selected_algos, show_pareto=show_pareto)


if __name__ == "__main__":
    print("Bienvenue dans l'outil de visualisation des benchmarks ANN!")
    interactive_visualization()