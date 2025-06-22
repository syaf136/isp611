import streamlit as st
import pandas as pd
import numpy as np
import random

# --- ACO PARAMETERS ---
st.title("📍 Campus Navigation Optimizer using ACO")

uploaded_file = st.file_uploader("Upload Distance Matrix CSV (in km)", type=["csv"])

if uploaded_file:
    distance_matrix = pd.read_csv(uploaded_file, index_col=0)
    nodes = list(distance_matrix.index)
    n_nodes = len(nodes)

    st.success(f"Loaded distance matrix with {n_nodes} buildings.")

    # User input for ACO
    start_node = st.selectbox("Start from", nodes, index=0)
    n_ants = st.slider("Number of Ants", 5, 50, 10)
    n_iterations = st.slider("Number of Iterations", 10, 100, 50)
    alpha = st.slider("Alpha (pheromone importance)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (heuristic importance)", 0.1, 5.0, 2.0)
    evaporation = st.slider("Pheromone evaporation rate", 0.0, 1.0, 0.5)

    if st.button("Run ACO Optimization"):
        dist = distance_matrix.values
        pheromone = np.ones((n_nodes, n_nodes))
        best_cost = float("inf")
        best_path = []

        def select_next_node(visited, current):
            probabilities = []
            for j in range(n_nodes):
                if j not in visited:
                    tau = pheromone[current][j] ** alpha
                    eta = (1.0 / dist[current][j]) ** beta if dist[current][j] > 0 else 0
                    probabilities.append(tau * eta)
                else:
                    probabilities.append(0)
            total = sum(probabilities)
            probabilities = [p / total if total > 0 else 0 for p in probabilities]
            return np.random.choice(range(n_nodes), p=probabilities)

        for iteration in range(n_iterations):
            all_paths = []
            all_costs = []
            for ant in range(n_ants):
                path = [nodes.index(start_node)]
                while len(path) < n_nodes:
                    next_node = select_next_node(path, path[-1])
                    path.append(next_node)
                path.append(path[0])  # return to start
                cost = sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))
                all_paths.append(path)
                all_costs.append(cost)

                if cost < best_cost:
                    best_cost = cost
                    best_path = path

            # Update pheromone
            pheromone *= (1 - evaporation)
            for path, cost in zip(all_paths, all_costs):
                for i in range(len(path) - 1):
                    pheromone[path[i]][path[i + 1]] += 1.0 / cost

        best_named_path = [nodes[i] for i in best_path]
        st.success("Best Path Found:")
        st.write(" → ".join(best_named_path))
        st.write(f"Total Distance: {round(best_cost, 3)} km")
