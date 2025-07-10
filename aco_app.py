import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import random

# --- Load Predefined Files ---
@st.cache_data
def load_data():
    dist_matrix = pd.read_csv("uitm_distance_matrix_km.csv", index_col=0)
    coords = pd.read_csv("uitm_named_buildings.csv")
    coords_dict = dict(zip(coords["name"], zip(coords["lat"], coords["lon"])))
    return dist_matrix, coords_dict, list(dist_matrix.index)

# --- ACO Algorithm ---
def run_aco(distance_matrix, nodes, start_node, end_node, n_ants, n_iterations, alpha, beta, evaporation, pheromone_constant, seed=None, early_stopping=False, max_no_improve=20):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_nodes = len(nodes)
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
        if total > 0:
            probabilities = [p / total for p in probabilities]
            return np.random.choice(range(n_nodes), p=probabilities)
        else:
            # Fallback: pick random unvisited node
            unvisited = [j for j in range(n_nodes) if j not in visited]
            return random.choice(unvisited) if unvisited else current

    no_improve_count = 0

    for _ in range(n_iterations):
        for _ in range(n_ants):
            start_index = nodes.index(start_node)
            end_index = nodes.index(end_node)
            path = [start_index]
            while path[-1] != end_index:
                next_node = select_next_node(path, path[-1])
                if next_node in path:
                    break
                path.append(next_node)
            if path[-1] != end_index:
                continue
            cost = sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))
            if cost < best_cost:
                best_cost = cost
                best_path = path
                no_improve_count = 0
            else:
                no_improve_count += 1
        pheromone *= (1 - evaporation)
        for i in range(len(best_path) - 1):
            pheromone[best_path[i]][best_path[i + 1]] += pheromone_constant / best_cost

        if early_stopping and no_improve_count >= max_no_improve:
            break

    return [nodes[i] for i in best_path], round(best_cost, 3)

# --- UI ---
st.set_page_config(layout="wide")
st.title("📍 UiTM Shah Alam Navigation Optimizer")

# Load data
distance_matrix, coords_dict, node_list = load_data()

# Inputs
start_node = st.selectbox("🏁 Start Location", node_list, index=0)
end_node = st.selectbox("🏁 Target Location", node_list, index=1)

with st.expander("⚙ ACO Parameters Settings"):
    n_ants = st.slider("Number of Ants", 5, 50, 10)
    n_iterations = st.slider("Iterations", 10, 200, 50)
    alpha = st.slider("Alpha (Pheromone Influence)", 0.1, 5.0, 1.0)
    beta = st.slider("Beta (Heuristic Influence)", 0.1, 5.0, 2.0)
    evaporation = st.slider("Evaporation Rate", 0.0, 1.0, 0.5)
    pheromone_constant = st.slider("Pheromone Constant (Q)", 10, 500, 100)
    run_multiple = st.slider("Run ACO N Times (Stable Best Path)", 1, 10, 1)
    seed_option = st.checkbox("Fix random seed (Stable output)", value=True)
    early_stop = st.checkbox("Enable early stopping if no improvement", value=True)

if st.button("Find Path"):
    best_overall_path = None
    best_overall_cost = float("inf")

    for i in range(run_multiple):
        seed = 42 + i if seed_option else None
        best_path, best_cost = run_aco(distance_matrix, node_list, start_node, end_node,
            n_ants, n_iterations, alpha, beta, evaporation, pheromone_constant,
            seed=seed, early_stopping=early_stop)

        if best_cost < best_overall_cost:
            best_overall_cost = best_cost
            best_overall_path = best_path

    if best_overall_path:
        st.success(f"✅ Best Path Found: {' → '.join(best_overall_path)}")
        st.markdown(f"*Total Distance:* {best_overall_cost} km")

        # Optional: breakdown
        with st.expander("📏 Path Breakdown"):
            total = 0
            for i in range(len(best_overall_path) - 1):
                a = best_overall_path[i]
                b = best_overall_path[i + 1]
                d = distance_matrix.loc[a, b]
                total += d
                st.write(f"➡ {a} to {b} = {round(d, 3)} km")
            st.markdown(f"*Total:* {round(total, 3)} km")

        # Draw path
        path_coords = [coords_dict[name] for name in best_overall_path]
        line_data = pd.DataFrame({
            "from_lat": [path_coords[i][0] for i in range(len(path_coords)-1)],
            "from_lon": [path_coords[i][1] for i in range(len(path_coords)-1)],
            "to_lat": [path_coords[i+1][0] for i in range(len(path_coords)-1)],
            "to_lon": [path_coords[i+1][1] for i in range(len(path_coords)-1)],
        })

        marker_data = pd.DataFrame([
            {"lat": lat, "lon": lon, "name": name}
            for name, (lat, lon) in coords_dict.items()
        ])

        line_layer = pdk.Layer("LineLayer", data=line_data,
            get_source_position='[from_lon, from_lat]',
            get_target_position='[to_lon, to_lat]',
            get_width=5, get_color=[255, 0, 0])

        dot_layer = pdk.Layer("ScatterplotLayer", data=marker_data,
            get_position='[lon, lat]', get_radius=6, get_fill_color=[0, 0, 255])

        text_layer = pdk.Layer("TextLayer", data=marker_data,
            get_position='[lon, lat]', get_text='name', get_size=16, get_color=[0, 0, 0])

        mid_lat, mid_lon = path_coords[0]
        st.pydeck_chart(pdk.Deck(
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            initial_view_state=pdk.ViewState(latitude=mid_lat, longitude=mid_lon, zoom=17),
            layers=[line_layer, dot_layer, text_layer],
            tooltip={"text": "{name}"}
        ))
    else:
        st.error("❌ No valid path was found. Please adjust parameters or try again.")
