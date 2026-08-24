import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertKnowledge, ILPSearch
from pgmpy.metrics import SHD, AdjacencyConfusionMatrix


def generate_synthetic_sem(n_nodes=10, edge_prob=0.25, n_samples=500, noise_scale=1.0, seed=42):
    """
    Generate continuous linear SEM observational dataset (Erdős-Rényi DAG)
    matching Manzour et al. (2021) Layered Network (LN) experimental setup.
    """
    rng = np.random.default_rng(seed)

    # 1. Erdős-Rényi DAG topology (topologically sorted)
    G_raw = nx.gnp_random_graph(n_nodes, edge_prob, directed=True, seed=seed)
    adj = np.triu(nx.to_numpy_array(G_raw), k=1)

    # 2. Edge weights sampled uniformly from [0.1, 1.0]
    weights = rng.uniform(0.1, 1.0, size=adj.shape)
    W = adj * weights

    # 3. Continuous dataset simulation: X = Noise * (I - W)^(-1)
    noise = rng.normal(0, noise_scale, size=(n_samples, n_nodes))
    I = np.eye(n_nodes)
    X = noise @ np.linalg.inv(I - W)

    col_names = [f"X{i}" for i in range(n_nodes)]
    df = pd.DataFrame(X, columns=col_names)
    # df = (df - df.mean()) / df.std()

    # Ground truth DAG
    true_dag = DAG()
    true_dag.add_nodes_from(col_names)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if W[i, j] != 0:
                true_dag.add_edge(f"X{i}", f"X{j}")

    return df, true_dag


def evaluate_ilp_config(config_name, penalty, l_penalty, expert_knowledge, df, true_dag, options=None):
    """
    Run Layered Network (LN) ILPSearch for a specific penalty, lambda, and search space setting.
    """
    t0 = time.time()
    try:
        ilp = ILPSearch(penalty=penalty, l_penalty=l_penalty, expert_knowledge=expert_knowledge, options=options)
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
        ub = milp_res.fun
        lb = milp_res.mip_dual_bound
        gap = milp_res.mip_gap
        is_timeout = milp_res.status == 3

        return {
            "Configuration": config_name,
            "Penalty": penalty.upper(),
            "Lambda": l_penalty,
            "Time (s)": round(elapsed, 4),
            "Timeout": is_timeout,
            "Best UB": round(ub, 4) if not np.isnan(ub) else "-",
            "Best LB": round(lb, 4) if not np.isnan(lb) else "-",
            "MIP Gap (%)": round(gap * 100, 2) if not np.isnan(gap) else "-",
            "SHD": int(shd_val),
            "TPR": round(tpr, 4),
            "FNR": round(fnr, 4),
            "FPR": round(fpr, 4),
            "Precision": round(precision, 4),
            "F1": round(f1, 4),
        }
    except Exception as e:
        return {
            "Configuration": config_name,
            "Penalty": penalty.upper(),
            "Lambda": l_penalty,
            "Time (s)": -1.0,
            "Timeout": False,
            "Best UB": "-",
            "Best LB": "-",
            "MIP Gap (%)": "-",
            "SHD": -1,
            "TPR": 0.0,
            "FNR": 0.0,
            "FPR": 0.0,
            "Precision": 0.0,
            "F1": 0.0,
            "Error": str(e),
        }


def run_mixture_benchmark(edge_prob=None, seed=42, l_penalties=None):
    if l_penalties is None:
        l_penalties = [0.05, 0.1, 1]

    m_values = [10, 20, 30, 40]
    n_values = [100, 1000]
    n_seeds = 5

    all_results = []

    print("=" * 120)
    print("8-GRAPH MIXTURE BENCHMARK (LN ILP SEARCH)")
    print(f"m in {m_values} | n in {n_values} | Lambdas = {l_penalties} | Seeds = {n_seeds}")
    print("=" * 120)

    for m in m_values:
        prob = edge_prob if edge_prob is not None else (2.0 / (m - 1) if m > 1 else 0.2)
        for n in n_values:
            print(f"\n---> Dataset: m={m} nodes, n={n} samples (edge_prob={prob:.4f})")

            for s_idx in range(n_seeds):
                current_seed = seed + s_idx
                df, true_dag = generate_synthetic_sem(n_nodes=m, edge_prob=prob, n_samples=n, seed=current_seed)

                # Moral Graph Expert Knowledge
                moral_graph = true_dag.moralize()
                moral_edges = []
                for u, v in moral_graph.edges():
                    moral_edges.append((u, v))
                    moral_edges.append((v, u))
                ek_moral = ExpertKnowledge(search_space=moral_edges)

                # Test L0 penalty for Complete vs Moral across each lambda
                test_cases = [
                    ("Complete", "l0", None),
                    ("Moral", "l0", ek_moral),
                ]

                for lp in l_penalties:
                    for case_name, pen, ek in test_cases:
                        print(f"  [Seed {current_seed}] Running {case_name} (penalty={pen}, lambda={lp})...")
                        res = evaluate_ilp_config(
                            config_name=case_name,
                            penalty=pen,
                            l_penalty=lp,
                            expert_knowledge=ek,
                            df=df,
                            true_dag=true_dag,
                            options={"time_limit": 50.0 * m},
                        )
                        res["m"] = m
                        res["n"] = n
                        all_results.append(res)

    results_df = pd.DataFrame(all_results)

    # Pre-process columns for aggregation: set strings "-" to NaN for averaging
    for col in ["Best UB", "Best LB", "MIP Gap (%)"]:
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    # Aggregate by averaging across seeds
    agg_cols = [
        "Time (s)",
        "Timeout",
        "Best UB",
        "Best LB",
        "MIP Gap (%)",
        "SHD",
        "TPR",
        "FNR",
        "FPR",
        "Precision",
        "F1",
    ]
    grouped_df = results_df.groupby(["Configuration", "Penalty", "Lambda", "m", "n"])[agg_cols].mean().reset_index()

    # Reorder columns
    cols = [
        "Configuration",
        "Penalty",
        "Lambda",
        "m",
        "n",
        "Time (s)",
        "Timeout",
        "Best UB",
        "Best LB",
        "MIP Gap (%)",
        "SHD",
        "TPR",
        "FNR",
        "FPR",
        "Precision",
        "F1",
    ]
    grouped_df = grouped_df[cols]

    print("\n" + "=" * 120)
    print("8-GRAPH MIXTURE BENCHMARK SUMMARY TABLES (AVERAGED OVER SEEDS)")
    print("=" * 120)

    # Group by Configuration, Penalty, Lambda
    groups = grouped_df.groupby(["Configuration", "Penalty", "Lambda"], sort=False)
    for (config_name, pen_type, lam_val), sub_df in groups:
        sub_print = sub_df.sort_values(by=["n", "m"]).drop(columns=["Configuration", "Penalty", "Lambda"])

        # Round averages for display and cast SHD to int
        sub_print = sub_print.round(4)
        sub_print["SHD"] = sub_print["SHD"].astype(int)

        print("\n" + "-" * 110)
        print(f"  SEARCH SPACE: {config_name.upper()} | PENALTY: {pen_type.upper()} | LAMBDA = {lam_val}")
        print("-" * 110)
        print(sub_print.to_string(index=False))
        print("-" * 110)

    print("=" * 120)

    return grouped_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 8-graph mixture benchmark with exact ILP search.")
    parser.add_argument("--prob", type=float, default=None, help="Erdos-Renyi edge probability (default: dynamic d=2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--lambda-pen", type=float, nargs="+", default=[0.05, 0.1, 1], help="Regularization penalty lambda(s)"
    )

    args = parser.parse_args()

    run_mixture_benchmark(edge_prob=args.prob, seed=args.seed, l_penalties=args.lambda_pen)
