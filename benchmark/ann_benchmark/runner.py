import importlib
from ann_benchmark.datasets import load_dataset
from ann_benchmark.evaluation import evaluate
import os
import json

def run_benchmark(config):
    dataset_name = config['dataset']
    xb, xq, gt = load_dataset(dataset_name)
    k = config['k']

    for algo_conf in config['algorithms']:
        module = importlib.import_module(algo_conf['module'])
        cls = getattr(module, algo_conf['class'])
        
        param_grid = expand_grid(algo_conf['parameters'])

        for params in param_grid:
            algo = cls(**params)
            algo.fit(xb)
            I = algo.query(xq, k)
            
            metrics = evaluate(I, gt, k)
            metrics["search_time"] = algo.last_search_time / len(xq)  # temps moyen par requête

            save_result(dataset_name, algo_conf['name'], params, metrics)

def expand_grid(param_dict):
    from itertools import product
    keys = list(param_dict.keys())
    values = [param_dict[k] for k in keys]
    combinations = product(*values)
    return [dict(zip(keys, comb)) for comb in combinations]


def save_result(dataset, algo_name, params, metrics):
    output_path = os.path.join("results", dataset, f"{algo_name}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    record = {
        "algorithm": algo_name,
        "parameters": params,
        "metrics": metrics
    }

    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)