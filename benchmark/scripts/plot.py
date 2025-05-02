import os
import json
import numpy as np
import plotly.graph_objects as go

def plot_all_methods(result_dir="/Volumes/SSD/M1VMI/S2/big_data/results"):
    """
    Génère une visualisation des résultats de benchmark avec une frontière de Pareto stricte.
    Affiche les points optimaux (non dominés) connectés par une ligne, avec tous les points
    en arrière-plan semi-transparent.
    """
    all_results = []

    # Chargement des résultats depuis les fichiers JSON
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

    # Regroupement par algorithme
    method_groups = {}
    for res in all_results:
        method = res["algorithm"]
        method_groups.setdefault(method, []).append(res)

    # Création du graphique
    fig = go.Figure()

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

        # Si nous avons assez de points, calculer la frontière de Pareto stricte
        if len(recall_values) >= 2:
            # Identifier les points Pareto-optimaux (non dominés)
            pareto_indices = []
            for i in range(len(recall_values)):
                is_dominated = False
                for j in range(len(recall_values)):
                    if i != j:
                        # Un point est dominé si un autre point a un recall >= et un qps >=
                        # avec au moins un des deux strictement supérieur
                        if ((recall_values[j] >= recall_values[i] and qps_values[j] > qps_values[i]) or
                            (recall_values[j] > recall_values[i] and qps_values[j] >= qps_values[i])):
                            is_dominated = True
                            break
                if not is_dominated:
                    pareto_indices.append(i)

            # Trier les points par recall croissant
            pareto_indices.sort(key=lambda i: recall_values[i])

            # Extraire les valeurs pour les points Pareto-optimaux
            pareto_recalls = [recall_values[i] for i in pareto_indices]
            pareto_qps = [qps_values[i] for i in pareto_indices]
            pareto_texts = [hover_texts[i] for i in pareto_indices]
            
            # Tracer la frontière de Pareto
            fig.add_trace(go.Scatter(
                x=pareto_recalls,
                y=pareto_qps,
                mode='lines+markers',
                name=method,
                hovertext=pareto_texts,
                hoverinfo='text',
                line=dict(width=2),
                marker=dict(size=8)
            ))
            
            # Ajouter tous les points en arrière-plan avec transparence
            fig.add_trace(go.Scatter(
                x=recall_values,
                y=qps_values,
                mode='markers',
                marker=dict(size=4, color='rgba(150,150,150,0.3)'),
                name=method + " (all)",
                hovertext=hover_texts,
                hoverinfo='text',
                showlegend=False
            ))
            
        else:
            # Pour les algorithmes avec peu de points, afficher tous les points
            fig.add_trace(go.Scatter(
                x=recall_values,
                y=qps_values,
                mode='lines+markers',
                name=method,
                hovertext=hover_texts,
                hoverinfo='text',
                line_shape='spline'
            ))

    # Configuration du layout
    fig.update_layout(
        title="Recall vs Queries per second - Frontière de Pareto",
        xaxis_title="Recall@10",
        yaxis_title="Queries per second (log scale)",
        yaxis_type="log",
        xaxis=dict(range=[0.0, 1.0]),
        hovermode="closest",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=1000,
        height=600
    )

    # Ajouter une grille pour faciliter la lecture
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.2)')

    # Affichage du graphique
    fig.show()
    
    # Option pour sauvegarder le graphique en HTML interactif
    # fig.write_html("pareto_benchmark_results.html")

if __name__ == "__main__":
    plot_all_methods()