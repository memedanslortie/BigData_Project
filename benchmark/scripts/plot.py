import os
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import re

def plot_all_datasets(base_dir="/Volumes/SSD/M1VMI/S2/big_data/results", output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and 
                any(f.endswith('.json') for f in os.listdir(os.path.join(base_dir, d)))]
    
    if not datasets:
        print(f"aucun dataset trouvé dans {base_dir}")
        return

    print(f"génération des visualisations pour {len(datasets)} datasets...")
    
    color_palette = px.colors.qualitative.D3
    
    all_methods = set()
    for dataset in datasets:
        result_dir = os.path.join(base_dir, dataset)
        for file in os.listdir(result_dir):
            if file.endswith(".json"):
                method_name = file.replace('.json', '')
                all_methods.add(method_name)
    
    method_colors = {method: color_palette[i % len(color_palette)] 
                     for i, method in enumerate(sorted(all_methods))}
    
    all_data = []
    
    for dataset in datasets:
        result_dir = os.path.join(base_dir, dataset)
        all_results = []
        
        for file in os.listdir(result_dir):
            if file.endswith(".json"):
                with open(os.path.join(result_dir, file)) as f:
                    try:
                        content = json.load(f)
                        if isinstance(content, list):
                            for item in content:
                                item["_source_file"] = file.replace('.json', '')
                            all_results.extend(content)
                    except Exception as e:
                        print(f"erreur dans {dataset}/{file}: {e}")
        
        if not all_results:
            print(f"aucun résultat pour {dataset}")
            continue
            
        method_groups = {}
        for res in all_results:
            method = res.get("_source_file", res.get("algorithm", "unknown"))
            method_groups.setdefault(method, []).append(res)
            
            all_data.append({
                "Dataset": prettify_dataset_name(dataset),
                "Algorithm": method,
                "Recall@10": res["metrics"].get("recall@10", 0),
                "QPS": 1.0 / max(res["metrics"].get("search_time", 1), 1e-9),
                "Build Time": res["metrics"].get("index_time", 0),
                "Parameters": ", ".join(f"{k}={v}" for k, v in res.get("parameters", {}).items())
            })
        
        fig = go.Figure()
        
        for method, res_list in method_groups.items():
            if len(res_list) == 0:
                continue
                
            if method not in method_colors:
                method_colors[method] = color_palette[len(method_colors) % len(color_palette)]
            
            color = method_colors[method]
            recall_values = []
            qps_values = []
            build_times = []
            hover_texts = []
            
            for res in sorted(res_list, key=lambda x: x["metrics"].get("recall@10", 0)):
                recall = res["metrics"].get("recall@10", 0)
                qps = 1.0 / max(res["metrics"].get("search_time", 1), 1e-9)
                build_time = res["metrics"].get("index_time", 0)
                
                param_str = ""
                if "parameters" in res:
                    param_items = []
                    for k, v in res["parameters"].items():
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            if abs(v) >= 1000:
                                param_items.append(f"{k}: {v:,}")
                            else:
                                param_items.append(f"{k}: {v}")
                        else:
                            param_items.append(f"{k}: {v}")
                    param_str = "<br>".join(param_items)
                
                hover_text = (
                    f"<b>{method}</b><br>"
                    f"<b>recall@10:</b> {recall:.4f}<br>"
                    f"<b>qps:</b> {qps:.2f}<br>"
                    f"<b>build time:</b> {build_time:.2f}s<br>"
                    f"<b>parameters:</b><br>{param_str}"
                )
                
                recall_values.append(recall)
                qps_values.append(qps)
                build_times.append(build_time)
                hover_texts.append(hover_text)
            
            if len(recall_values) >= 2:
                pareto_indices = compute_pareto_frontier(recall_values, qps_values)
                
                pareto_indices.sort(key=lambda i: recall_values[i])
                
                pareto_recalls = [recall_values[i] for i in pareto_indices]
                pareto_qps = [qps_values[i] for i in pareto_indices]
                pareto_texts = [hover_texts[i] for i in pareto_indices]
                
                fig.add_trace(
                    go.Scatter(
                        x=recall_values,
                        y=qps_values,
                        mode='markers',
                        marker=dict(
                            size=7, 
                            color=color, 
                            opacity=0.1,
                            line=dict(width=1, color='rgba(0,0,0,0.2)')
                        ),
                        name=method + " (all)",
                        hovertext=hover_texts,
                        hoverinfo='text',
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pareto_recalls,
                        y=pareto_qps,
                        mode='lines+markers',
                        name=method,
                        hovertext=pareto_texts,
                        hoverinfo='text',
                        line=dict(width=3, color=color),
                        marker=dict(
                            size=9, 
                            color=color,
                            symbol='circle',
                            line=dict(width=1.5, color='white')
                        )
                    )
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=recall_values,
                        y=qps_values,
                        mode='markers',
                        name=method,
                        hovertext=hover_texts,
                        hoverinfo='text',
                        marker=dict(
                            size=10, 
                            color=color,
                            symbol='circle',
                            line=dict(width=1.5, color='white')
                        )
                    )
                )
        
        fig.update_xaxes(
            title_text="recall@10",
            range=[0, 1.05],
            tickformat='.2f'
        )
        fig.update_yaxes(
            title_text="query per second (QPS)",
            type="log",
            tickformat='.1e'
        )
        
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220,220,220,0.25)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(220,220,220,0.25)')
    
        dataset_title = prettify_dataset_name(dataset)
        fig.update_layout(
            title={
                'text': f"QPS vs Recall: {dataset_title}",
                'font': {'size': 24, 'family': 'arial, sans-serif', 'color': '#333333'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=800,
            width=1000,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                font=dict(size=12),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            hovermode="closest",
            template="plotly_white",
            margin=dict(l=80, r=50, t=120, b=80),
            plot_bgcolor='rgba(250,250,252,1)',
            paper_bgcolor='rgba(250,250,252,1)',
            font=dict(family="arial, sans-serif", size=12, color="#333333")
        )
        
        # Générer un fichier PNG séparé pour chaque dataset
        dataset_png = os.path.join(output_dir, f"{dataset}_pareto_comparison.png")
        fig.write_image(dataset_png, scale=2)
        
        # Générer également une version HTML pour l'interactivité
        dataset_html = os.path.join(output_dir, f"{dataset}_pareto_comparison.html")
        fig.write_html(dataset_html)
        
        print(f"visualisation pour {dataset} sauvegardée dans: {dataset_png} et {dataset_html}")
    
    df = pd.DataFrame(all_data)
    
    if not df.empty:
        best_configs = pd.DataFrame()
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            for algo in dataset_df['Algorithm'].unique():
                algo_df = dataset_df[dataset_df['Algorithm'] == algo]
                if not algo_df.empty:
                    max_recall = algo_df['Recall@10'].max()
                    threshold = max_recall * 0.50
                    qualified = algo_df[algo_df['Recall@10'] >= threshold]
                    if not qualified.empty:
                        best_row = qualified.loc[qualified['QPS'].idxmax()]
                        best_configs = pd.concat([best_configs, pd.DataFrame([best_row])], ignore_index=True)
        
        radar_metrics = {}
        
        for dataset in df['Dataset'].unique():
            dataset_df = best_configs[best_configs['Dataset'] == dataset]
            
            max_recall = dataset_df['Recall@10'].max() if not dataset_df.empty else 1
            max_qps = dataset_df['QPS'].max() if not dataset_df.empty else 1
            min_build = dataset_df['Build Time'].min() if not dataset_df.empty else 0.1
            
            for _, row in dataset_df.iterrows():
                algo = row['Algorithm']
                if algo not in radar_metrics:
                    radar_metrics[algo] = {
                        'Recall': [],
                        'Speed': [],
                        'Build Time': []
                    }
                
                norm_recall = row['Recall@10'] / max_recall if max_recall > 0 else 0
                norm_speed = row['QPS'] / max_qps if max_qps > 0 else 0
                norm_build = min_build / row['Build Time'] if row['Build Time'] > 0 else 1
                
                radar_metrics[algo]['Recall'].append(norm_recall)
                radar_metrics[algo]['Speed'].append(norm_speed)
                radar_metrics[algo]['Build Time'].append(norm_build)
        
        radar_data = []
        for algo, metrics in radar_metrics.items():
            radar_data.append({
                'Algorithm': algo,
                'Recall': np.mean(metrics['Recall']),
                'Speed': np.mean(metrics['Speed']),
                'Build Time': np.mean(metrics['Build Time'])
            })
        
        radar_df = pd.DataFrame(radar_data)
        
        if not radar_df.empty:
            radar_categories = ['Recall', 'Speed', 'Build Time']
            
            fig_radar = go.Figure()
            
            for i, row in radar_df.iterrows():
                algo = row['Algorithm']
                values = [row['Recall'], row['Speed'], row['Build Time']]
                
                values_closed = values + [values[0]]
                categories_closed = radar_categories + [radar_categories[0]]
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=categories_closed,
                    fill='toself',
                    name=algo,
                    line_color=method_colors.get(algo, 'gray'),
                    fillcolor=adjust_opacity(method_colors.get(algo, 'gray'), 0.2)
                ))
            
            fig_radar.update_layout(
                title={
                    'text': "comparaison globale des algorithmes ann",
                    'font': {'size': 22, 'family': 'arial, sans-serif'},
                    'x': 0.5,
                    'xanchor': 'center'
                },
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.1,
                    xanchor="center",
                    x=0.5
                ),
                width=800,
                height=700,
                template="plotly_white"
            )
            
            radar_png = os.path.join(output_dir, "algo_radar_comparison.png")
            fig_radar.write_image(radar_png, scale=2)
            
            radar_html = os.path.join(output_dir, "algo_radar_comparison.html")
            fig_radar.write_html(radar_html)
            
            print(f"graphique radar sauvegardé dans: {radar_png} et {radar_html}")
    
    generate_summary_html(base_dir, output_dir)
    
    print("génération des visualisations terminée!")

def generate_summary_html(base_dir="/Volumes/SSD/M1VMI/S2/big_data/results", output_dir="visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    datasets = [d for d in os.listdir(base_dir) 
                if os.path.isdir(os.path.join(base_dir, d)) and 
                any(f.endswith('.json') for f in os.listdir(os.path.join(base_dir, d)))]
    
    for dataset in datasets:
        result_dir = os.path.join(base_dir, dataset)
        
        for file in os.listdir(result_dir):
            if file.endswith(".json"):
                algo_name = file.replace('.json', '')
                
                with open(os.path.join(result_dir, file)) as f:
                    try:
                        results = json.load(f)
                        
                        if isinstance(results, list):
                            for res in results:
                                recall = res["metrics"].get("recall@10", 0)
                                qps = 1.0 / max(res["metrics"].get("search_time", 1), 1e-9)
                                build_time = res["metrics"].get("index_time", 0)
                                
                                params = {}
                                if "parameters" in res:
                                    params = res["parameters"]
                                
                                all_data.append({
                                    "Dataset": dataset,
                                    "Algorithm": algo_name,
                                    "Recall@10": recall,
                                    "QPS": qps,
                                    "Build Time": build_time,
                                    "Parameters": ", ".join(f"{k}={v}" for k, v in params.items())
                                })
                    except Exception as e:
                        print(f"erreur de lecture du fichier {file}: {e}")
    
    df = pd.DataFrame(all_data)
    
    if df.empty:
        print("aucune donnée trouvée pour générer le rapport.")
        return
    
    summary_data = []
    
    for dataset in df["Dataset"].unique():
        ds_data = df[df["Dataset"] == dataset].copy()
        
        best_recall_idx = ds_data["Recall@10"].idxmax()
        worst_recall_idx = ds_data["Recall@10"].idxmin()
        
        best_recall = ds_data.loc[best_recall_idx].copy()
        worst_recall = ds_data.loc[worst_recall_idx].copy()
        
        best_qps_idx = ds_data["QPS"].idxmax()
        worst_qps_idx = ds_data["QPS"].idxmin()
        
        best_qps = ds_data.loc[best_qps_idx].copy()
        worst_qps = ds_data.loc[worst_qps_idx].copy()
        
        best_build_idx = ds_data["Build Time"].idxmin()
        worst_build_idx = ds_data["Build Time"].idxmax()
        
        best_build = ds_data.loc[best_build_idx].copy()
        worst_build = ds_data.loc[worst_build_idx].copy()
        
        ds_data.loc[:, "score"] = ds_data["Recall@10"] * np.log10(ds_data["QPS"] + 1)
        best_balance_idx = ds_data["score"].idxmax()
        best_balance = ds_data.loc[best_balance_idx].copy()
        
        best_recall_value = f"{best_recall['Recall@10']:.4f}"
        worst_recall_value = f"{worst_recall['Recall@10']:.4f}"
        best_qps_value = f"{best_qps['QPS']:.2f} qps"
        worst_qps_value = f"{worst_qps['QPS']:.2f} qps"
        best_build_value = f"{best_build['Build Time']:.2f} secondes"
        worst_build_value = f"{worst_build['Build Time']:.2f} secondes"
        best_balance_value = f"recall={best_balance['Recall@10']:.4f}, qps={best_balance['QPS']:.2f}"
        
        summary_data.extend([
            {
                "Dataset": dataset,
                "Metric": "meilleur rappel",
                "Algorithm": best_recall["Algorithm"],
                "Value": best_recall_value,
                "Parameters": best_recall["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "pire rappel",
                "Algorithm": worst_recall["Algorithm"],
                "Value": worst_recall_value,
                "Parameters": worst_recall["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "meilleure vitesse de requête",
                "Algorithm": best_qps["Algorithm"],
                "Value": best_qps_value,
                "Parameters": best_qps["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "pire vitesse de requête",
                "Algorithm": worst_qps["Algorithm"],
                "Value": worst_qps_value,
                "Parameters": worst_qps["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "construction d'index la plus rapide",
                "Algorithm": best_build["Algorithm"],
                "Value": best_build_value,
                "Parameters": best_build["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "construction d'index la plus lente",
                "Algorithm": worst_build["Algorithm"],
                "Value": worst_build_value,
                "Parameters": worst_build["Parameters"]
            },
            {
                "Dataset": dataset,
                "Metric": "meilleur compromis rappel/vitesse",
                "Algorithm": best_balance["Algorithm"],
                "Value": best_balance_value,
                "Parameters": best_balance["Parameters"]
            }
        ])
    
    summary_df = pd.DataFrame(summary_data)
    
    algo_stats = []
    for algo in df["Algorithm"].unique():
        algo_data = df[df["Algorithm"] == algo]
        
        algo_stats.append({
            "Algorithm": algo,
            "Avg Recall@10": algo_data["Recall@10"].mean(),
            "Max Recall@10": algo_data["Recall@10"].max(),
            "Avg QPS": algo_data["QPS"].mean(),
            "Max QPS": algo_data["QPS"].max(),
            "Avg Build Time": algo_data["Build Time"].mean(),
            "Min Build Time": algo_data["Build Time"].min()
        })
    
    algo_stats_df = pd.DataFrame(algo_stats)
    
    html_content = """
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>résumé des performances des algorithmes ann</title>
        <style>
            body {
                font-family: arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f7f9;
            }
            
            h1, h2, h3 {
                color: #2c3e50;
                margin-top: 30px;
                margin-bottom: 15px;
            }
            
            h1 {
                text-align: center;
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
            }
            
            .summary-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                grid-gap: 20px;
                margin-bottom: 40px;
            }
            
            .dataset-card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            .dataset-name {
                color: white;
                background-color: #3498db;
                padding: 10px;
                margin: -20px -20px 20px -20px;
                border-radius: 8px 8px 0 0;
                font-size: 18px;
                font-weight: bold;
            }
            
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
                background-color: white;
                box-shadow: 0 2px 3px rgba(0, 0, 0, 0.1);
            }
            
            th, td {
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #e1e1e1;
            }
            
            th {
                background-color: #f8f9fa;
                color: #333;
                font-weight: bold;
            }
            
            tr:hover {
                background-color: #f1f9ff;
            }
            
            .best {
                background-color: #d4edda;
            }
            
            .worst {
                background-color: #f8d7da;
            }
            
            .balanced {
                background-color: #fff3cd;
            }
            
            .metric-name {
                font-weight: bold;
            }
            
            .algorithm-name {
                font-weight: bold;
                color: #1a365d;
            }
            
            .parameters {
                font-size: 0.85em;
                color: #555;
                max-width: 280px;
                word-wrap: break-word;
            }
            
            .comparison-table {
                margin-top: 30px;
                width: 100%;
                border-collapse: collapse;
            }
            
            .comparison-table th {
                background-color: #2c3e50;
                color: white;
                position: sticky;
                top: 0;
            }
            
            .nav-container {
                position: sticky;
                top: 0;
                background-color: rgba(255, 255, 255, 0.95);
                padding: 10px 0;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                z-index: 1000;
            }
            
            .nav-links {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 15px;
                padding: 0;
                margin: 0;
                list-style: none;
            }
            
            .nav-links li a {
                display: block;
                padding: 8px 15px;
                background-color: #3498db;
                color: white;
                text-decoration: none;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            
            .nav-links li a:hover {
                background-color: #2980b9;
            }
            
            .section {
                margin-top: 40px;
                padding-top: 15px;
            }
            
            .highlight-box {
                background-color: #e8f4f8;
                border-left: 5px solid #3498db;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 4px 4px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            
            .text-center {
                text-align: center;
            }
            
            footer {
                margin-top: 50px;
                text-align: center;
                padding: 20px;
                font-size: 0.8em;
                color: #7f8c8d;
                border-top: 1px solid #eee;
            }
        </style>
    </head>
    <body>
        <h1>résumé des performances des algorithmes ann</h1>
        
        <div class="nav-container">
            <ul class="nav-links">
                <li><a href="#performance-par-dataset">performance par dataset</a></li>
                <li><a href="#comparaison-algorithmes">comparaison des algorithmes</a></li>
            </ul>
        </div>
        
        <section id="performance-par-dataset" class="section">
            <h2>performance par dataset</h2>
            <p>cette section montre les algorithmes les plus performants pour chaque dataset selon différentes métriques.</p>
            
            <div class="summary-container">
    """
    
    for dataset in summary_df["Dataset"].unique():
        dataset_summary = summary_df[summary_df["Dataset"] == dataset]
        
        html_content += f"""
            <div class="dataset-card">
                <div class="dataset-name">{dataset}</div>
                <table>
                    <thead>
                        <tr>
                            <th>métrique</th>
                            <th>algorithme</th>
                            <th>valeur</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for _, row in dataset_summary.iterrows():
            css_class = ""
            if "meilleur" in row["Metric"] and "construction" not in row["Metric"]:
                css_class = "best"
            elif "pire" in row["Metric"]:
                css_class = "worst"
            elif "compromis" in row["Metric"]:
                css_class = "balanced"
                
            html_content += f"""
                <tr class="{css_class}">
                    <td class="metric-name">{row["Metric"]}</td>
                    <td class="algorithm-name">{row["Algorithm"]}</td>
                    <td>{row["Value"]}</td>
                </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
        """
    
    html_content += """
            </div>
        </section>
        
        <section id="comparaison-algorithmes" class="section">
            <h2>comparaison globale des algorithmes</h2>
            <p>ce tableau compare les performances moyennes et maximales de chaque algorithme sur tous les datasets.</p>
            
            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>algorithme</th>
                        <th>rappel moyen</th>
                        <th>rappel max</th>
                        <th>qps moyen</th>
                        <th>qps max</th>
                        <th>temps de construction moyen</th>
                        <th>temps de construction min</th>
                    </tr>
                </thead>
                <tbody>
    """
    
    for _, row in algo_stats_df.iterrows():
        html_content += f"""
            <tr>
                <td class="algorithm-name">{row["Algorithm"]}</td>
                <td>{row["Avg Recall@10"]:.4f}</td>
                <td>{row["Max Recall@10"]:.4f}</td>
                <td>{row["Avg QPS"]:.2f}</td>
                <td>{row["Max QPS"]:.2f}</td>
                <td>{row["Avg Build Time"]:.2f}s</td>
                <td>{row["Min Build Time"]:.2f}s</td>
            </tr>
        """
    
    html_content += """
                </tbody>
            </table>
        </section>
    </body>
    </html>
    """
    
    output_file = os.path.join(output_dir, "best_configurations.html")
    with open(output_file, "w") as f:
        f.write(html_content)
    
    print(f"rapport de synthèse sauvegardé dans: {output_file}")
    return output_file

def compute_pareto_frontier(recalls, qps_values):
    pareto_indices = []
    n = len(recalls)
    
    for i in range(n):
        is_dominated = False
        for j in range(n):
            if i != j:
                if ((recalls[j] >= recalls[i] and qps_values[j] > qps_values[i]) or
                    (recalls[j] > recalls[i] and qps_values[j] >= qps_values[i])):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_indices.append(i)
    
    return pareto_indices

def adjust_opacity(color, opacity):
    if color.startswith('#'):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f'rgba({r},{g},{b},{opacity})'
    
    if color.startswith('rgba'):
        parts = color.split(',')
        parts[-1] = f'{opacity})'
        return ','.join(parts)
    
    if color.startswith('rgb'):
        rgb = color.replace('rgb(', '').replace(')', '')
        return f'rgba({rgb},{opacity})'
    
    return color

def prettify_dataset_name(name):
    name = re.sub(r'\.(hdf5|h5)', '', name)
    
    if 'lastfm' in name.lower():
        if 'dot' in name.lower():
            return "last.fm (dot product)"
        else:
            return "last.fm (angular)"
    
    if 'nytimes' in name.lower():
        return f"new york times ({name.split('-')[1]}d, angular)"
    
    if 'glove' in name.lower():
        return f"glove word vectors ({name.split('-')[1]}d)"
    
    parts = name.split('-')
    if len(parts) >= 2 and parts[-1].lower() in ['angular', 'euclidean', 'dot']:
        metric = parts[-1].capitalize()
        rest = '-'.join(parts[:-1])
        dim_part = next((part for part in parts[:-1] if part.isdigit()), None)
        if dim_part:
            rest = rest.replace(f"-{dim_part}", f" ({dim_part}d)")
        return f"{rest.capitalize()}, {metric}"
    
    return name.capitalize()

if __name__ == "__main__":
    plot_all_datasets()