import argparse
import time

import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge
from pgmpy.causal_discovery.ILPSearch import ILPSearch
from pgmpy.metrics import SHD, AdjacencyConfusionMatrix


def generate_synthetic_sem(n_nodes=10, edge_prob=0.25, n_samples=500, noise_scale=1.0, seed=42):
    """
    Generate continuous linear SEM observational dataset (Erdős-Rényi DAG)
    matching Manzour et al. (2021) Layered Network (LN) experimental setup.
    """
    rng = np.random.default_rng(seed)

    # 1. Erdős-Rényi DAG topology (topologically sorted)
    W = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                W[i, j] = rng.uniform(0.5, 2.0) * rng.choice([-1, 1])

    # 2. Generate continuous data
    X = np.zeros((n_samples, n_nodes))
    for j in range(n_nodes):
        noise = rng.normal(0, noise_scale, size=n_samples)
        if j == 0:
            X[:, j] = noise
        else:
            X[:, j] = X[:, :j] @ W[:j, j] + noise

    df = pd.DataFrame(X, columns=[f"X{i}" for i in range(n_nodes)])

    # Create Ground Truth DAG
    true_dag = DAG()
    true_dag.add_nodes_from(df.columns)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if W[i, j] != 0:
                true_dag.add_edge(f"X{i}", f"X{j}")

    return df, true_dag


def evaluate_ilp_config(config_name, l_penalty, expert_knowledge, df, true_dag, options=None):
    """
    Run Layered Network (LN) ILPSearch for a specific penalty, lambda, and search space setting.
    """
    t0 = time.time()
    try:
        ilp = ILPSearch(l_penalty=l_penalty, expert_knowledge=expert_knowledge, options=options)
        ilp.fit(df)
        elapsed = time.time() - t0

        learned_dag = ilp.causal_graph_

        # Graph Metrics
        shd_val = SHD()(true_dag, learned_dag)
        adj_cm = AdjacencyConfusionMatrix()(true_dag, learned_dag)

        tpr = adj_cm.get("recall", 0.0)
        fnr = 1.0 - tpr
        fpr = 1.0 - adj_cm.get("specificity", 1.0)
        precision = adj_cm.get("precision", 0.0)
        f1 = adj_cm.get("f1", 0.0)

        # SciPy MILP Solver Optimization Stats directly from fitted instance
        milp_res = ilp.milp_result_

        if milp_res is not None:
            best_ub = milp_res.fun
            best_lb = getattr(milp_res, "mip_dual_bound", best_ub)
            mip_gap = getattr(milp_res, "mip_gap", 0.0)
        else:
            best_ub, best_lb, mip_gap = float("nan"), float("nan"), float("nan")

        return {
            "Configuration": config_name,
            "Penalty": "L0",
            "Lambda": l_penalty,
            "Time (s)": round(elapsed, 4),
            "Timeout": 1 if elapsed >= options.get("time_limit", 999999) else 0,
            "Best UB": round(best_ub, 4),
            "Best LB": round(best_lb, 4),
            "MIP Gap (%)": round(mip_gap * 100, 2),
            "SHD": shd_val,
            "TPR": round(tpr, 4),
            "FNR": round(fnr, 4),
            "FPR": round(fpr, 4),
            "Precision": round(precision, 4),
            "F1": round(f1, 4),
        }
    except Exception:
        import traceback

        traceback.print_exc()
        return {
            "Configuration": config_name,
            "Penalty": "L0",
            "Lambda": l_penalty,
            "Time (s)": -1.0,
            "Timeout": 0.0,
            "Best UB": float("nan"),
            "Best LB": float("nan"),
            "MIP Gap (%)": float("nan"),
            "SHD": -1,
            "TPR": 0.0,
            "FNR": 0.0,
            "FPR": 0.0,
            "Precision": 0.0,
            "F1": 0.0,
        }


def run_mixture_benchmark(edge_prob=0.25, seed=42, l_penalties=None):
    if l_penalties is None:
        l_penalties = [0.1, 1.0]

    print("=" * 120)
    print("8-GRAPH MIXTURE BENCHMARK (LN ILP SEARCH)")
    print(f"m in [10, 20, 30, 40] | n in [100, 1000] | Lambdas = {l_penalties} | Seeds = 1")
    print("=" * 120)

    all_results = []

    # We only use seed=42 as requested
    current_seed = seed

    for n in [100, 1000]:
        for m in [10, 20, 30, 40]:
            ep = 2.0 / (m - 1)
            print(f"\n---> Dataset: n={n} samples, m={m} nodes (edge_prob={ep:.4f})")

            df, true_dag = generate_synthetic_sem(
                n_nodes=m,
                edge_prob=ep,
                n_samples=n,
                noise_scale=1.0,
                seed=current_seed,
            )

            # Define the "Moral" Search Space precisely
            moral_graph = true_dag.moralize()
            moral_edges = []
            for u, v in moral_graph.edges():
                moral_edges.append((u, v))
                moral_edges.append((v, u))
            ek_moral = ExpertKnowledge(search_space=moral_edges)

            # Test L0 penalty for Complete vs Moral across each lambda
            test_cases = [
                ("Complete", None),
                ("Moral", ek_moral),
            ]

            for lp in l_penalties:
                for case_name, ek in test_cases:
                    print(f"  [Seed {current_seed}] Running {case_name} (lambda={lp})...")
                    res = evaluate_ilp_config(
                        config_name=case_name,
                        l_penalty=lp,
                        expert_knowledge=ek,
                        df=df,
                        true_dag=true_dag,
                        options={"time_limit": 600},
                    )
                    res["m"] = m
                    res["n"] = n
                    all_results.append(res)

    results_df = pd.DataFrame(all_results)

    print("\n" + "=" * 120)
    print("8-GRAPH MIXTURE BENCHMARK SUMMARY TABLES (AVERAGED OVER SEEDS)")
    print("=" * 120)

    # Group by Configuration, Penalty, and Lambda
    for lp in l_penalties:
        for config in ["Complete", "Moral"]:
            subset = results_df[
                (results_df["Configuration"] == config) & (results_df["Penalty"] == "L0") & (results_df["Lambda"] == lp)
            ]

            if subset.empty:
                continue

            summary = subset.groupby(["n", "m"]).mean(numeric_only=True).reset_index()
            summary = summary.drop(columns=["Lambda"])

            print(f"\n{'-' * 110}")
            print(f"  SEARCH SPACE: {config.upper()} | PENALTY: L0 | LAMBDA = {lp}")
            print(f"{'-' * 110}")
            print(summary.to_string(index=False))
            print(f"{'-' * 110}")

    print("=" * 120 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-pen", type=float, nargs="+", default=[0.1, 1.0])
    args = parser.parse_args()

    run_mixture_benchmark(edge_prob=args.prob, seed=args.seed, l_penalties=args.lambda_pen)
