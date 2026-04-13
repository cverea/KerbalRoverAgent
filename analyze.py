import pandas as pd
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIGURACIÓN
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(current_dir, "Experimento 5")

csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

SUCCESS_THRESHOLD_M = 5.0  # criterio de éxito
OUTPUT_DIR = os.path.join(current_dir, "results_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# FUNCIONES AUXILIARES
# =========================
def compute_episode_metrics(df: pd.DataFrame, file_path: str) -> dict:
    df = df.sort_values("step").reset_index(drop=True)

    first_row = df.iloc[0]
    last_row = df.iloc[-1]

    initial_distance_m = float(first_row["distance_m"])
    final_distance_m = float(last_row["distance_m"])
    
    n_steps = int(last_row["step"]) + 1
    success = n_steps < 100
    episode_duration_s = float(last_row["timestamp"] - first_row["timestamp"])

    mean_speed_mps = float(df["speed_mps"].mean())
    max_speed_mps = float(df["speed_mps"].max())

    n_drive_actions = int((df["tool"] == "drive").sum())
    n_brake_actions = int((df["tool"] == "brake").sum())

    brake_rows = df[df["tool"] == "brake"]
    total_brake_time_s = float(brake_rows["duration_s"].sum())
    mean_brake_strength = float(brake_rows["brake_strength"].mean()) if len(brake_rows) > 0 else 0.0

    n_safety_overrides = int(df["overridden_by_safety"].sum())
    pct_safety_overrides = 100.0 * n_safety_overrides / len(df)

    mean_abs_bearing_deg = float(df["bearing_deg"].abs().mean())
    max_abs_bearing_deg = float(df["bearing_deg"].abs().max())

    distance_reduction_m = initial_distance_m - final_distance_m
    progress_ratio = float(distance_reduction_m / initial_distance_m) if initial_distance_m > 0 else np.nan

    steer_diff_mean = float(df["steer"].diff().abs().mean())
    throttle_diff_mean = float(df["throttle"].diff().abs().mean())

    # Cambios de signo en steer como aproximación a zig-zag
    steer_nonzero = df["steer"].replace(0, np.nan).dropna()
    if len(steer_nonzero) > 1:
        steer_sign_changes = int((np.sign(steer_nonzero).diff().fillna(0) != 0).sum() - 1)
        steer_sign_changes = max(0, steer_sign_changes)
    else:
        steer_sign_changes = 0

    # Cambios de signo en forward_m como aproximación a overshooting
    forward_sign = np.sign(df["forward_m"])
    if len(forward_sign) > 1:
        forward_sign_changes = int((forward_sign.diff().fillna(0) != 0).sum() - 1)
        forward_sign_changes = max(0, forward_sign_changes)
    else:
        forward_sign_changes = 0

    return {
        "file": os.path.basename(file_path),
        "success": success,
        "initial_distance_m": initial_distance_m,
        "final_distance_m": final_distance_m,
        "distance_reduction_m": distance_reduction_m,
        "progress_ratio": progress_ratio,
        "n_steps": n_steps,
        "episode_duration_s": episode_duration_s,
        "mean_speed_mps": mean_speed_mps,
        "max_speed_mps": max_speed_mps,
        "n_drive_actions": n_drive_actions,
        "n_brake_actions": n_brake_actions,
        "total_brake_time_s": total_brake_time_s,
        "mean_brake_strength": mean_brake_strength,
        "n_safety_overrides": n_safety_overrides,
        "pct_safety_overrides": pct_safety_overrides,
        "mean_abs_bearing_deg": mean_abs_bearing_deg,
        "max_abs_bearing_deg": max_abs_bearing_deg,
        "steer_diff_mean": steer_diff_mean,
        "throttle_diff_mean": throttle_diff_mean,
        "steer_sign_changes": steer_sign_changes,
        "forward_sign_changes": forward_sign_changes,
    }


def save_latex_table(summary_df: pd.DataFrame, filepath: str) -> None:
    latex = summary_df.to_latex(index=False, float_format="%.2f")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex)


# =========================
# CARGA Y MÉTRICAS
# =========================
episode_metrics = []
all_runs = {}

for file in csv_files:
    df = pd.read_csv(file)
    if df.empty:
        continue
    df = df.sort_values("step").reset_index(drop=True)
    all_runs[os.path.basename(file)] = df
    episode_metrics.append(compute_episode_metrics(df, file))

metrics_df = pd.DataFrame(episode_metrics)

# Guardar métricas por episodio
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "episode_metrics_summary.csv"), index=False)

# =========================
# TABLA FINAL 
# =========================
summary_table = pd.DataFrame([{
    "N episodios": len(metrics_df),
    "Tasa de éxito (%)": 100 * metrics_df["success"].mean(),
    "Distancia final media (m)": metrics_df["final_distance_m"].mean(),
    "Distancia final std (m)": metrics_df["final_distance_m"].std(),
    "Pasos medios": metrics_df["n_steps"].mean(),
    "Pasos std": metrics_df["n_steps"].std(),
    "Duración media (s)": metrics_df["episode_duration_s"].mean(),
    "Velocidad media (m/s)": metrics_df["mean_speed_mps"].mean(),
    "Velocidad máxima media (m/s)": metrics_df["max_speed_mps"].mean(),
    "Frenadas medias": metrics_df["n_brake_actions"].mean(),
    "Safety overrides medios": metrics_df["n_safety_overrides"].mean(),
    "Bearing abs medio (deg)": metrics_df["mean_abs_bearing_deg"].mean(),
    "Cambios de signo steer medios": metrics_df["steer_sign_changes"].mean(),
}])

summary_csv = os.path.join(OUTPUT_DIR, "final_summary_table.csv")
summary_table.to_csv(summary_csv, index=False)

summary_latex = os.path.join(OUTPUT_DIR, "final_summary_table.tex")
save_latex_table(summary_table, summary_latex)

print("Tabla resumen guardada en:")
print(summary_csv)
print(summary_latex)

# =========================
# HISTOGRAMAS
# =========================
hist_metrics = [
    ("final_distance_m", "Distancia final (m)", "hist_final_distance.png"),
    ("n_steps", "Número de pasos", "hist_steps.png"),
    ("mean_speed_mps", "Velocidad media (m/s)", "hist_mean_speed.png"),
    ("n_brake_actions", "Número de frenadas", "hist_brakes.png"),
]

for col, xlabel, filename in hist_metrics:
    plt.figure(figsize=(7, 5))
    plt.hist(metrics_df[col], bins=10)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de {xlabel}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()

print("Histogramas guardados.")

# =========================
# DISTANCIA VS PASO
# =========================
plt.figure(figsize=(8, 5))

for file_name, df in all_runs.items():
    plt.plot(df["step"], df["distance_m"], alpha=0.35)

plt.xlabel("Paso")
plt.ylabel("Distancia al objetivo (m)")
plt.title("Distancia al objetivo frente al número de pasos")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "distance_vs_step_all_runs.png"), dpi=300)
plt.close()

print("Gráfica distance_vs_step_all_runs.png guardada.")


