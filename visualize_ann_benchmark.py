#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def parse_args():
    """Parse les arguments pour configurer la visualisation."""
    parser = argparse.ArgumentParser(description='Visualisation des résultats de benchmark ANN')
    
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Dossier contenant les résultats du benchmark')
    
    parser.add_argument('--dataset', type=str, required=True,
                        help='Nom du dataset à visualiser')
    
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                        help='Algorithmes à inclure dans la visualisation')
    
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder l\'image générée')
    
    parser.add_argument('--style', type=str, default='darkgrid',
                        choices=['darkgrid', 'whitegrid', 'dark', 'white', 'ticks'],
                        help='Style de fond pour seaborn')
    
    parser.add_argument('--palette', type=str, default='colorblind',
                        choices=['colorblind', 'deep', 'muted', 'bright', 'pastel', 'dark'],
                        help='Palette de couleurs à utiliser')
    
    parser.add_argument('--metric', type=str, default='recall@10',
                        help='Métrique de précision à utiliser')
    
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Transparence des marqueurs')
    
    parser.add_argument('--fig-width', type=float, default=12,
                        help='Largeur de la figure en pouces')
    
    parser.add_argument('--fig-height', type=float, default=8,
                        help='Hauteur de la figure en pouces')
    
    parser.add_argument('--include-params', action='store_true', default=False,
                        help='Inclure les paramètres clés dans les étiquettes')
    
    parser.add_argument('--highlight-best', action='store_true', default=False,
                        help='Mettre en évidence les configurations optimales')
    
    parser.add_argument('--show-pareto', action='store_true', default=False,
                        help='Montrer la frontière de Pareto')
    
    return parser.parse_args()

def load_results(results_dir, dataset, algorithms=None):
    """Charge les résultats de benchmark depuis les fichiers JSON."""
    dataset_dir = os.path.join(results_dir, dataset)
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dossier de résultats non trouvé: {dataset_dir}")
    
    # Collecter tous les résultats
    all_results = []
    
    for file in os.listdir(dataset_dir):
        if not file.endswith('.json'):
            continue
            
        algo_name = file.split('.')[0]
        if algorithms and algo_name not in algorithms:
            continue
            
        try:
            with open(os.path.join(dataset_dir, file), 'r') as f:
                results = json.load(f)
                
                if isinstance(results, list):
                    # Ajouter l'algorithme si pas déjà présent dans chaque élément
                    for res in results:
                        if 'algorithm' not in res:
                            res['algorithm'] = algo_name
                    all_results.extend(results)
        except Exception as e:
            print(f"Erreur lors de la lecture de {file}: {e}")
    
    return all_results

def prepare_dataframe(results, recall_metric='recall@10'):
    """Convertit les résultats en DataFrame pandas pour faciliter l'analyse."""
    records = []
    
    for res in results:
        # Extraire les informations principales
        algo = res['algorithm']
        params = res['parameters']
        metrics = res['metrics']
        
        # Créer un enregistrement plat
        record = {
            'algorithm': algo,
            'recall': metrics.get(recall_metric, 0),
            'qps': 1.0 / metrics.get('search_time', float('inf')),
            'index_time': metrics.get('index_time', 0),
        }
        
        # Ajouter les paramètres principaux selon l'algorithme
        if algo == 'hnsw':
            record.update({
                'M': params.get('M', None),
                'ef_construction': params.get('ef_construction', None),
                'ef': params.get('ef', None),
                'param_str': f"M={params.get('M', '?')}, ef={params.get('ef', '?')}"
            })
        elif algo == 'annoy':
            record.update({
                'n_trees': params.get('n_trees', None),
                'search_k': params.get('search_k', None),
                'param_str': f"n_trees={params.get('n_trees', '?')}, search_k={params.get('search_k', '?')}"
            })
        elif algo == 'faiss_flat':
            record['param_str'] = "exact search"
        elif algo == 'faiss_ivf':
            record.update({
                'nlist': params.get('nlist', None),
                'nprobe': params.get('nprobe', None),
                'param_str': f"nlist={params.get('nlist', '?')}, nprobe={params.get('nprobe', '?')}"
            })
        else:
            # Paramètres génériques pour les autres algorithmes
            param_str = ", ".join(f"{k}={v}" for k, v in params.items() 
                                if k in ['M', 'ef', 'n_trees', 'search_k', 'nlist', 'nprobe'])
            record['param_str'] = param_str if param_str else "default"
            
        records.append(record)
    
    return pd.DataFrame(records)

def find_pareto_frontier(df, x_col='recall', y_col='qps'):
    """Trouve la frontière de Pareto pour un DataFrame."""
    df_sorted = df.sort_values(by=x_col)
    pareto_points = []
    current_best_y = 0
    
    for _, row in df_sorted.iterrows():
        if row[y_col] > current_best_y:
            pareto_points.append(row)
            current_best_y = row[y_col]
    
    return pd.DataFrame(pareto_points)

def visualize(df, args):
    """Génère une visualisation avancée avec Seaborn."""
    # Configurer le style
    sns.set(style=args.style)
    sns.set_palette(args.palette)
    
    # Créer la figure
    plt.figure(figsize=(args.fig_width, args.fig_height))
    
    # Plot principal
    ax = plt.subplot(111)
    
    # Grouper par algorithme pour les couleurs
    for algo, group in df.groupby('algorithm'):
        # Tracer les points et les lignes
        sns.scatterplot(
            data=group,
            x='recall',
            y='qps',
            label=algo,
            s=100,
            alpha=args.alpha,
            ax=ax
        )
        
        # Ajouter des lignes entre les points du même algorithme
        sorted_group = group.sort_values(by='recall')
        plt.plot(
            sorted_group['recall'], 
            sorted_group['qps'],
            alpha=0.5, 
            linewidth=1.5
        )
    
    # Tracer la frontière de Pareto si demandé
    if args.show_pareto and len(df['algorithm'].unique()) > 1:
        pareto_df = find_pareto_frontier(df, 'recall', 'qps')
        plt.plot(
            pareto_df['recall'],
            pareto_df['qps'],
            '--',
            color='black',
            linewidth=1.5,
            label='Frontière de Pareto'
        )
    
    # Mettre en évidence les meilleures configurations si demandé
    if args.highlight_best:
        # Meilleur pour chaque algorithme par recall
        best_recall = df.loc[df.groupby('algorithm')['recall'].idxmax()]
        
        # Meilleur pour chaque algorithme par QPS
        best_qps = df.loc[df.groupby('algorithm')['qps'].idxmax()]
        
        # Meilleur compromis (heuristique simple: recall * log(qps))
        df['score'] = df['recall'] * np.log(df['qps'])
        best_compromise = df.loc[df.groupby('algorithm')['score'].idxmax()]
        
        # Tracer ces points
        plt.scatter(
            best_recall['recall'], 
            best_recall['qps'], 
            s=150, 
            color='gold', 
            edgecolors='black', 
            zorder=10, 
            label='Meilleur recall'
        )
        
        plt.scatter(
            best_qps['recall'], 
            best_qps['qps'], 
            s=150, 
            color='lightgreen', 
            edgecolors='black', 
            zorder=10, 
            label='Meilleur QPS'
        )
        
        plt.scatter(
            best_compromise['recall'], 
            best_compromise['qps'], 
            s=150, 
            color='red', 
            marker='*', 
            edgecolors='black', 
            zorder=10, 
            label='Meilleur compromis'
        )
    
    # Ajouter des étiquettes pour les points si demandé
    if args.include_params:
        for i, row in df.iterrows():
            if row['algorithm'] == 'hnsw' and row['recall'] > 0.98:  # Exemple de filtre
                plt.annotate(
                    row['param_str'],
                    (row['recall'], row['qps']),
                    fontsize=8,
                    alpha=0.7,
                    xytext=(5, 5),
                    textcoords='offset points'
                )
    
    # Configuration esthétique
    plt.title(f"Performance des algorithmes ANN sur {args.dataset}", fontsize=16)
    plt.xlabel(f"Precision ({args.metric})", fontsize=14)
    plt.ylabel("Requêtes par seconde (QPS)", fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlim(max(0, df['recall'].min() - 0.05), 1.01)
    
    # Ajouter des lignes de grille pour les niveaux de recall courants
    for recall_level in [0.95, 0.99, 0.999]:
        plt.axvline(x=recall_level, linestyle='--', color='gray', alpha=0.5)
        plt.text(
            recall_level + 0.001, 
            df['qps'].min() * 1.1,
            f"{recall_level:.3f}",
            rotation=90, 
            verticalalignment='bottom',
            alpha=0.7
        )
    
    # Légende
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
    
    # Ajustement de la mise en page
    plt.tight_layout()
    
    # Sauvegarder si un chemin est spécifié
    if args.output:
        plt.savefig(args.output, dpi=300, bbox_inches='tight')
        print(f"Visualisation sauvegardée dans {args.output}")
    
    # Afficher
    plt.show()

def add_efficiency_metrics(df):
    """Ajoute des métriques d'efficacité supplémentaires."""
    # Efficacité = recall / log(temps de construction)
    df['build_efficiency'] = df['recall'] / np.log1p(df['index_time'])
    
    # Efficacité temporelle = recall * log(QPS)
    df['time_efficiency'] = df['recall'] * np.log(df['qps'])
    
    # Score global = combinaison pondérée
    df['overall_score'] = 0.7 * df['recall'] * np.log(df['qps']) + 0.3 * df['recall'] / np.log1p(df['index_time'])
    
    return df

def print_summary(df):
    """Imprime un résumé des résultats pour chaque algorithme."""
    print("\n===== RÉSUMÉ DES PERFORMANCES =====")
    
    for algo, group in df.groupby('algorithm'):
        print(f"\n{algo.upper()}:")
        
        # Configuration avec le meilleur recall
        best_recall_idx = group['recall'].idxmax()
        best_recall = group.loc[best_recall_idx]
        
        # Configuration avec le meilleur QPS
        best_qps_idx = group['qps'].idxmax()
        best_qps = group.loc[best_qps_idx]
        
        # Meilleur compromis (selon score)
        best_overall_idx = group['overall_score'].idxmax()
        best_overall = group.loc[best_overall_idx]
        
        print(f"  Meilleur recall: {best_recall['recall']:.4f} (QPS: {best_recall['qps']:.2f}, params: {best_recall['param_str']})")
        print(f"  Meilleur QPS: {best_qps['qps']:.2f} (recall: {best_qps['recall']:.4f}, params: {best_qps['param_str']})")
        print(f"  Meilleur compromis: recall={best_overall['recall']:.4f}, QPS={best_overall['qps']:.2f}, params: {best_overall['param_str']}")
        
        print(f"  Plage de recall: {group['recall'].min():.4f} - {group['recall'].max():.4f}")
        print(f"  Plage de QPS: {group['qps'].min():.2f} - {group['qps'].max():.2f}")
        print(f"  Temps d'indexation: {group['index_time'].min():.2f}s - {group['index_time'].max():.2f}s")

def main():
    """Fonction principale."""
    args = parse_args()
    
    try:
        results = load_results(args.results_dir, args.dataset, args.algorithms)
        
        if not results:
            print("Aucun résultat trouvé.")
            return

        df = prepare_dataframe(results, args.metric)
        df = add_efficiency_metrics(df)
        
        print_summary(df)
        visualize(df, args)
        
    except Exception as e:
        print(f"Erreur: {e}")

if __name__ == "__main__":
    main()


# a la racine : python visualize_ann_benchmark.py --dataset fashion-mnist-784-euclidean --algorithms hnsw annoy --highlight-best