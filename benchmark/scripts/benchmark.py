import os
import json
import time
import argparse
import yaml
import numpy as np
import importlib
from collections import defaultdict
import matplotlib.pyplot as plt

# Import des fonctions utiles depuis le framework de benchmark existant
from benchmark.ann_benchmark.datasets import load_dataset
from benchmark.ann_benchmark.evaluation import evaluate

def parse_args():
    """Parse les arguments en ligne de commande pour configurer le benchmark."""
    parser = argparse.ArgumentParser(description='Benchmark pour les algorithmes ANN')
    
    parser.add_argument('--config', type=str, default='benchmark/benchmark/full_comparison.yaml',
                        help='Chemin vers le fichier de configuration YAML')
    
    parser.add_argument('--dataset', type=str, default=None,
                        help='Jeu de données à utiliser, écrase celui spécifié dans le fichier YAML')
    
    parser.add_argument('--algorithms', type=str, nargs='+', default=None,
                        help='Algorithmes à tester (ex: annoy faiss_ivfpq voyager hnsw)')
    
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Dossier où sauvegarder les résultats')
    
    parser.add_argument('--k', type=int, default=None,
                        help='Nombre de voisins à rechercher')
    
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Générer et afficher la visualisation après le benchmark')
    
    return parser.parse_args()

def load_config(config_path):
    """Charge la configuration depuis le fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Erreur lors du chargement de la configuration: {e}")
        return None

def filter_algorithms(config, selected_algos):
    """Filtre les algorithmes à tester en fonction de la sélection de l'utilisateur."""
    if not selected_algos:
        return config['algorithms']
        
    filtered_algos = []
    for algo in config['algorithms']:
        if algo['name'] in selected_algos:
            filtered_algos.append(algo)
            
    if not filtered_algos:
        print(f"Aucun des algorithmes spécifiés ({', '.join(selected_algos)}) n'a été trouvé dans la configuration.")
        return config['algorithms']
        
    return filtered_algos

def expand_grid(param_dict):
    """Génère toutes les combinaisons possibles de paramètres."""
    from itertools import product
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    combinations = product(*values)
    return [dict(zip(keys, comb)) for comb in combinations]

def save_result(output_dir, dataset, algo_name, params, metrics):
    """Sauvegarde les résultats du benchmark dans un fichier JSON."""
    output_path = os.path.join(output_dir, dataset, f"{algo_name}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record = {
        "algorithm": algo_name,
        "parameters": params,
        "metrics": metrics
    }

    # Charger les résultats existants s'ils existent
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Ajouter le nouveau résultat
    data.append(record)

    # Sauvegarder les résultats
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    return output_path

def visualize_results(output_dir, dataset):
    """Génère un graphique comparant les performances des différents algorithmes."""
    dataset_dir = os.path.join(output_dir, dataset)
    if not os.path.exists(dataset_dir):
        print(f"Dossier de résultats non trouvé: {dataset_dir}")
        return
        
    # Collecter tous les résultats
    all_results = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.json'):
            try:
                with open(os.path.join(dataset_dir, file), 'r') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                print(f"Erreur lors de la lecture de {file}: {e}")
                
    if not all_results:
        print("Aucun résultat trouvé à visualiser.")
        return
        
    # Organiser par algorithme
    method_groups = defaultdict(list)
    for res in all_results:
        method = res["algorithm"]
        method_groups[method].append(res)
        
    # Créer le graphique
    plt.figure(figsize=(10, 6))
    
    for method, res_list in method_groups.items():
        # Trier par recall
        res_list = sorted(res_list, key=lambda x: x["metrics"]["recall@10"])
        
        recalls = [res["metrics"]["recall@10"] for res in res_list]
        qps_values = [1.0 / res["metrics"]["search_time"] for res in res_list]
        
        plt.plot(recalls, qps_values, 'o-', label=method)
        
    plt.title(f"Recall vs QPS pour {dataset}")
    plt.xlabel("Recall@10")
    plt.ylabel("Queries per second (log scale)")
    plt.yscale('log')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{dataset}_performance.png"))
    plt.show()

def run_benchmark(config_path, args):
    """Exécute le benchmark selon la configuration spécifiée."""
    # Charger la configuration
    config = load_config(config_path)
    if not config:
        return
        
    # Appliquer les écrasements de la ligne de commande
    dataset_name = args.dataset or config['dataset']
    k = args.k or config['k']
    
    # Filtrer les algorithmes si nécessaire
    algorithms = filter_algorithms(config, args.algorithms)
    
    print(f"Exécution du benchmark pour {dataset_name} avec k={k}")
    print(f"Algorithmes à tester: {', '.join(algo['name'] for algo in algorithms)}")
    
    # Charger les données
    try:
        xb, xq, gt = load_dataset(dataset_name)
        print(f"Données chargées: train={xb.shape}, test={xq.shape}, ground truth={gt.shape}")
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        return
    
    # Exécuter chaque algorithme avec ses configurations
    for algo_conf in algorithms:
        print(f"\nTest de {algo_conf['name']}...")
        
        # Importer dynamiquement la classe d'algorithme
        try:
            module = importlib.import_module(algo_conf['module'])
            cls = getattr(module, algo_conf['class'])
        except (ImportError, AttributeError) as e:
            print(f"Erreur lors de l'importation de {algo_conf['module']}.{algo_conf['class']}: {e}")
            continue
        
        # Générer toutes les combinaisons de paramètres
        param_grid = expand_grid(algo_conf['parameters'])
        print(f"  {len(param_grid)} combinaisons de paramètres à tester")
        
        # Tester chaque combinaison de paramètres
        for i, params in enumerate(param_grid):
            print(f"  Configuration {i+1}/{len(param_grid)}: {params}")
            
            try:
                # Initialiser l'algorithme
                algo = cls(**params)
                
                # Construire l'index
                print("    Construction de l'index...")
                start_time = time.time()
                algo.fit(xb)
                index_time = time.time() - start_time
                print(f"    Index construit en {index_time:.2f}s")
                
                # Exécuter les requêtes
                print(f"    Exécution des requêtes pour k={k}...")
                I = algo.query(xq, k)
                
                # Évaluer les résultats
                metrics = evaluate(I, gt, k)
                metrics["search_time"] = algo.last_search_time / len(xq)  # temps moyen par requête
                metrics["index_time"] = index_time
                
                print(f"    Recall@{k}: {metrics[f'recall@{k}']:.4f}, "
                      f"QPS: {1.0/metrics['search_time']:.2f}, "
                      f"Index time: {metrics['index_time']:.2f}s")
                
                # Sauvegarder les résultats
                save_path = save_result(args.output_dir, dataset_name, algo_conf['name'], params, metrics)
                print(f"    Résultats sauvegardés dans {save_path}")
                
            except Exception as e:
                print(f"    Erreur lors du test de {algo_conf['name']} avec {params}: {e}")
    
    print("\nBenchmark terminé!")
    
    # Visualiser les résultats si demandé
    if args.visualize:
        print("Génération de la visualisation...")
        visualize_results(args.output_dir, dataset_name)

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.config, args)



    # a la racine : python -m benchmark.scripts.benchmark --config benchmark/benchmark/hnsw_comparison.yaml --algorithms hnsw --dataset fashion-mnist-784-euclidean --visualize