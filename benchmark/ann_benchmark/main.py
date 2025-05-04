import yaml
from ann_benchmark.runner import run_benchmark

if __name__ == "__main__":
    with open("benchmark/nmslib_comparison.yaml") as f:
        config = yaml.safe_load(f)

    run_benchmark(config)

# python -m ann_benchmark.main