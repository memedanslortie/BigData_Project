import os
import json
import time
import argparse
import yaml
import numpy as np
import importlib
from collections import defaultdict
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
import logging
from tqdm import tqdm
import faiss

from benchmark.ann_benchmark.datasets import load_dataset
from benchmark.ann_benchmark.evaluation import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    parser.add_argument('--distance', type=str, default=None,
                        help='Type de distance à utiliser (euclidean, angular, jaccard). Écrase celui du fichier YAML')
    
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='Générer et afficher la visualisation après le benchmark')
    
    parser.add_argument('--parallel', type=str, choices=['none', 'inner', 'outer', 'both'], default='both',
                       help='Type de parallélisation: none=aucune, inner=threads internes, outer=processes parallèles, both=les deux')
    
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Nombre maximum de workers pour la parallélisation externe (default: CPU count - 1)')
    
    parser.add_argument('--inner-threads', type=int, default=None,
                       help='Nombre de threads pour la parallélisation interne de chaque algo (default: dépend de parallel mode)')
    
    return parser.parse_args()

def load_config(config_path):
    """Charge la configuration depuis le fichier YAML."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Erreur lors du chargement de la configuration: {e}")
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
        logger.warning(f"Aucun des algorithmes spécifiés ({', '.join(selected_algos)}) n'a été trouvé dans la configuration.")
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

    import filelock
    lock_path = output_path + ".lock"
    
    with filelock.FileLock(lock_path):
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
        logger.warning(f"Dossier de résultats non trouvé: {dataset_dir}")
        return
        
    # Collecter tous les résultats
    all_results = []
    for file in os.listdir(dataset_dir):
        if file.endswith('.json') and not file.endswith('.lock'):
            try:
                with open(os.path.join(dataset_dir, file), 'r') as f:
                    results = json.load(f)
                    if isinstance(results, list):
                        all_results.extend(results)
            except Exception as e:
                logger.error(f"Erreur lors de la lecture de {file}: {e}")
                
    if not all_results:
        logger.warning("Aucun résultat trouvé à visualiser.")
        return
        
    # Organiser par algorithme
    method_groups = defaultdict(list)
    for res in all_results:
        method = res["algorithm"]
        method_groups[method].append(res)
        
    # Créer le graphique
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*']
    colors = plt.cm.tab10.colors
    
    for i, (method, res_list) in enumerate(method_groups.items()):
        # Trier par recall
        res_list = sorted(res_list, key=lambda x: x["metrics"].get("recall@10", 0))
        
        recalls = [res["metrics"].get("recall@10", 0) for res in res_list]
        qps_values = [1.0 / max(res["metrics"].get("search_time", 1e-6), 1e-6) for res in res_list]
        
        marker = markers[i % len(markers)]
        color = colors[i % len(colors)]
        plt.plot(recalls, qps_values, marker=marker, linestyle='-', color=color, label=method)
        
    plt.title(f"Performance Comparison - {dataset}")
    plt.xlabel("Recall@10")
    plt.ylabel("Queries per second (log scale)")
    plt.yscale('log')
    plt.xlim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{dataset}_performance.png")
    plt.savefig(output_path, dpi=300)
    logger.info(f"Graphique sauvegardé dans {output_path}")
    plt.close()

def detect_distance_type(dataset_name):
    """
    Détecte automatiquement le type de distance en fonction du nom du dataset.
    """
    dataset_name = dataset_name.lower()
    
    # Datasets avec distance angulaire
    if any(term in dataset_name for term in ['glove', 'angular', 'cosine', 'word2vec', 'text', 'nytimes', 
                                           'deep', 'last.fm', 'coco-i2i', 'coco-t2i']):
        return 'angular'
    
    # Datasets avec distance de Jaccard
    if any(term in dataset_name for term in ['jaccard', 'kosarak', 'movielens']):
        return 'jaccard'
    
    # Par défaut, on utilise la distance euclidienne
    return 'euclidean'

def add_metric_params(algo_conf, params, distance_type):
    """
    Ajoute le paramètre de métrique approprié à chaque algorithme en fonction
    de son implémentation.
    """
    algo_name = algo_conf['name'].lower()
    params_copy = params.copy()  # Créer une copie pour ne pas modifier l'original
    
    # Mappings pour les noms de paramètres de métrique par algorithme
    metric_param_names = {
        'annoy': 'metric',
        'faiss_ivfpq': 'metric',
        'faisshnw': 'metric',
        'faishnsw': 'metric',
        'hnsw': 'space',
        'voyager': 'metric',
        'qsg-ngt': 'metric'
    }
    
    # Mappings pour les valeurs de métrique par algorithme et type de distance
    metric_values = {
        'annoy': {
            'euclidean': 'euclidean',
            'angular': 'angular',
            'jaccard': 'hamming'  # Approximation
        },
        'faiss_ivfpq': {
            'euclidean': 'l2',
            'angular': 'cosine',
            'jaccard': 'l2'  # Non supporté directement
        },
        'faisshnw': {
            'euclidean': 'l2',
            'angular': 'cosine',
            'jaccard': 'l2'  # Non supporté directement
        },
        'faishnsw': {
            'euclidean': 'l2',
            'angular': 'cosine',
            'jaccard': 'l2'  # Non supporté directement
        },
        'hnsw': {
            'euclidean': 'l2',
            'angular': 'cosine',
            'jaccard': 'l2'  # Non supporté directement
        },
        'voyager': {
            'euclidean': 'euclidean',
            'angular': 'cosine',
            'jaccard': 'euclidean'  # Non supporté directement
        },
        'qsg-ngt': {
            'euclidean': 'L2',
            'angular': 'Cosine',
            'jaccard': 'Jaccard'
        }
    }
    
    # Trouver le nom d'algo qui correspond le mieux dans notre mapping
    matching_algo = None
    for key in metric_param_names.keys():
        if key in algo_name:
            matching_algo = key
            break
    
    # Si l'algo est supporté, ajouter le paramètre de métrique
    if matching_algo:
        param_name = metric_param_names[matching_algo]
        
        # Si l'algorithme a une configuration spécifique pour cette distance
        if matching_algo in metric_values and distance_type in metric_values[matching_algo]:
            metric_value = metric_values[matching_algo][distance_type]
            
            # Si ce paramètre n'est pas déjà défini dans les paramètres originaux
            if param_name not in params:
                params_copy[param_name] = metric_value
                logger.info(f"Ajout automatique de {param_name}={metric_value} pour {algo_name}")
    
    return params_copy

def execute_single_config(algo_conf, params, xb, xq, gt, k, output_dir, dataset_name, inner_threads, distance_type):
    """Exécute un benchmark pour une seule configuration d'algorithme."""
    algo_name = algo_conf['name']
    
    # Ajouter le paramètre de métrique si nécessaire
    params_with_metric = add_metric_params(algo_conf, params, distance_type)
    
    # Prétraitement des données en fonction de la distance
    if distance_type == 'angular' or distance_type == 'cosine':
        # Pour certains algorithmes sensibles aux problèmes de normalisation (notamment FAISS IVFPQ)
        if algo_name == 'faiss_ivfpq':
            # Forcer l'utilisation de L2 sur des vecteurs normalisés plutôt que IP directement
            params_with_metric['metric'] = 'l2'
            
    # Configurer la parallélisation interne si nécessaire
    if inner_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(inner_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(inner_threads)
        os.environ["MKL_NUM_THREADS"] = str(inner_threads)
        # Pour FAISS
        try:
            faiss.omp_set_num_threads(inner_threads)
        except:
            pass
    
    try:
        # Importer dynamiquement la classe d'algorithme
        module = importlib.import_module(algo_conf['module'])
        cls = getattr(module, algo_conf['class'])
        
        # Initialiser l'algorithme avec les paramètres et la métrique
        algo_instance = cls(**params_with_metric)
        
        # Ajouter le nombre de threads au logger pour hnswlib
        if hasattr(algo_instance, 'index') and hasattr(algo_instance.index, 'set_num_threads'):
            try:
                algo_instance.index.set_num_threads(inner_threads)
            except:
                pass
        
        # Pour la distance angulaire, on pré-normalise les vecteurs pour certains algorithmes problématiques
        if (distance_type == 'angular' or distance_type == 'cosine') and algo_name == 'faiss_ivfpq':
            xb_copy = xb.copy().astype(np.float32)
            xq_copy = xq.copy().astype(np.float32)
            faiss.normalize_L2(xb_copy)
            faiss.normalize_L2(xq_copy)
            
            # Construire l'index avec des vecteurs normalisés
            start_time = time.time()
            algo_instance.fit(xb_copy)
            index_time = time.time() - start_time
            
            # Exécuter les requêtes sur des vecteurs normalisés
            I = algo_instance.query(xq_copy, k)
        else:
            # Construire l'index
            start_time = time.time()
            algo_instance.fit(xb)
            index_time = time.time() - start_time
            
            # Exécuter les requêtes
            I = algo_instance.query(xq, k)
        
        # Évaluer les résultats
        metrics = evaluate(I, gt, k)
        metrics["search_time"] = algo_instance.last_search_time / len(xq)  # temps moyen par requête
        metrics["index_time"] = index_time
        metrics["qps"] = 1.0 / metrics["search_time"]
        
        # Sauvegarder les résultats
        save_path = save_result(output_dir, dataset_name, algo_conf['name'], params_with_metric, metrics)
        
        return {
            "algorithm": algo_name,
            "params": params_with_metric,
            "metrics": metrics,
            "save_path": save_path,
            "status": "success"
        }
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "algorithm": algo_name,
            "params": params_with_metric,
            "status": "error",
            "error_message": str(e),
            "traceback": tb
        }

def run_benchmark(config_path, args):
    """Exécute le benchmark selon la configuration spécifiée."""
    # Charger la configuration
    config = load_config(config_path)
    if not config:
        return
        
    # Appliquer les écrasements de la ligne de commande
    dataset_name = args.dataset or config['dataset']
    k = args.k or config['k']
    
    # Déterminer le type de distance à utiliser
    distance_type = args.distance or config.get('distance') or detect_distance_type(dataset_name)
    logger.info(f"Utilisation de la distance: {distance_type}")
    
    # Filtrer les algorithmes si nécessaire
    algorithms = filter_algorithms(config, args.algorithms)
    
    logger.info(f"Exécution du benchmark pour {dataset_name} avec k={k}")
    logger.info(f"Algorithmes à tester: {', '.join(algo['name'] for algo in algorithms)}")
    
    # Configuration de la parallélisation
    cpu_count = multiprocessing.cpu_count()
    if args.parallel == 'none':
        max_workers = 1
        inner_threads = 1
    elif args.parallel == 'inner':
        max_workers = 1
        inner_threads = args.inner_threads or cpu_count
    elif args.parallel == 'outer':
        max_workers = args.max_workers or (cpu_count - 1)
        inner_threads = 1
    else:  # 'both'
        max_workers = args.max_workers or max(1, cpu_count // 2)
        inner_threads = args.inner_threads or (cpu_count // max_workers)
    
    logger.info(f"Configuration de parallélisation: mode={args.parallel}, "
                f"max_workers={max_workers}, inner_threads={inner_threads}")
    
    # Charger les données
    try:
        xb, xq, gt = load_dataset(dataset_name)
        logger.info(f"Données chargées: train={xb.shape}, test={xq.shape}, ground truth={gt.shape}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return
    
    # Préparer toutes les configurations à tester
    all_configs = []
    for algo_conf in algorithms:
        param_grid = expand_grid(algo_conf['parameters'])
        logger.info(f"Algorithme {algo_conf['name']}: {len(param_grid)} combinaisons de paramètres")
        
        for params in param_grid:
            all_configs.append((algo_conf, params))
    
    total_configs = len(all_configs)
    logger.info(f"Total: {total_configs} configurations à tester")
    
    # Exécution en parallèle ou séquentielle selon le mode
    if args.parallel in ['outer', 'both'] and max_workers > 1:
        # Mode parallèle
        results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_config = {
                executor.submit(
                    execute_single_config, 
                    algo_conf, params, xb, xq, gt, k, args.output_dir, dataset_name, inner_threads, distance_type
                ): (algo_conf, params) 
                for algo_conf, params in all_configs
            }
            
            # Afficher la progression avec tqdm
            with tqdm(total=total_configs, desc="Configurations testées") as pbar:
                for future in concurrent.futures.as_completed(future_to_config):
                    algo_conf, params = future_to_config[future]
                    try:
                        result = future.result()
                        if result["status"] == "success":
                            logger.info(f"{algo_conf['name']} avec {params}: "
                                      f"Recall@{k}={result['metrics'][f'recall@{k}']:.4f}, "
                                      f"QPS={result['metrics']['qps']:.2f}")
                        else:
                            logger.error(f"{algo_conf['name']} avec {params}: ERREUR: {result['error_message']}")
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Exception non capturée: {e}")
                    finally:
                        pbar.update(1)
    else:
        # Mode séquentiel
        results = []
        for i, (algo_conf, params) in enumerate(all_configs):
            logger.info(f"Configuration {i+1}/{total_configs}: {algo_conf['name']} avec {params}")
            result = execute_single_config(algo_conf, params, xb, xq, gt, k, args.output_dir, dataset_name, inner_threads, distance_type)
            if result["status"] == "success":
                logger.info(f"  Recall@{k}: {result['metrics'][f'recall@{k}']:.4f}, "
                          f"QPS: {result['metrics']['qps']:.2f}, "
                          f"Index time: {result['metrics']['index_time']:.2f}s")
            else:
                logger.error(f"  ERREUR: {result['error_message']}")
            results.append(result)
    
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"\nBenchmark terminé! {success_count} succès, {error_count} échecs sur {total_configs} configurations")
    
    if args.visualize:
        logger.info("Génération de la visualisation...")
        visualize_results(args.output_dir, dataset_name)

if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()
    run_benchmark(args.config, args)
    elapsed = time.time() - start_time
    logger.info(f"Temps total d'exécution: {elapsed:.2f} secondes ({elapsed/3600:.2f} heures)")


    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset lastfm-64-dot --distance angular --parallel inner --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --visualize


    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms faiss_ivfpq --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms faissHSNW --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms hnsw --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms voyager --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset nytimes-256-angular --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms annoy --visualize
    # python -m benchmark.scripts.benchmark --config benchmark/benchmark/full_comparison.yaml --dataset lastfm-64-dot --distance angular --parallel both --max-workers 4 --inner-threads 2 --algorithms QSG-NGT --visualize