def calculate_35_node_grid_state(step_idx: int):
    # step_idx bude napr. 0, 1, 2, 3, 4, 5, 6, 7, 8 (každý krok = +6 hodín)
    hours_ahead = step_idx * 6
    dwd_raw = fetch_35_nodes_dwd()

    dwd_t = np.zeros((7, 5))
    dwd_wspd = np.zeros((7, 5))
    dwd_wdir = np.zeros((7, 5))
    dwd_prec = np.zeros((7, 5))
    dwd_cape = np.zeros((7, 5))
    dwd_dem = np.zeros((7, 5))

    if dwd_raw and isinstance(dwd_raw, list) and len(dwd_raw) == 35:
        # Pre 6-hodinové kroky berieme index priamo ako step_idx * 6 (alebo podla potreby)
        data_idx = min(max(0, step_idx * 6), 48)

        idx = 0
        for j in range(5):
            for i in range(7):
                n = dwd_raw[idx].get("hourly", {})
                dwd_t[i, j] = n.get("temperature_2m", [16.0])[data_idx] if len(n.get("temperature_2m", [])) > data_idx else 16.0
                dwd_wspd[i, j] = (n.get("wind_speed_10m", [15.0])[data_idx] / 3.6) if len(n.get("wind_speed_10m", [])) > data_idx else 4.0
                dwd_wdir[i, j] = n.get("wind_direction_10m", [315.0])[data_idx] if len(n.get("wind_direction_10m", [])) > data_idx else 315.0
                dwd_prec[i, j] = n.get("precipitation", [0.0])[data_idx] if len(n.get("precipitation", [])) > data_idx else 0.0
                dwd_cape[i, j] = n.get("cape", [0.0])[data_idx] if len(n.get("cape", [])) > data_idx else 0.0
                dwd_dem[i, j] = dwd_raw[idx].get("elevation", 1200.0)
                idx += 1
    else:
        for j in range(5):
            for i in range(7):
                dwd_dem[i, j] = 900.0 + j * 150.0
                dwd_t[i, j] = 18.0 - (dwd_dem[i, j] - 672.0) * 0.0065
                dwd_wspd[i, j] = 4.5
                dwd_wdir[i, j] = 315.0
                dwd_prec[i, j] = 0.0
                dwd_cape[i, j] = 0.0
