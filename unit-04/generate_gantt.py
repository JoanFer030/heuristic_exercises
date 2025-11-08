import matplotlib.pyplot as plt
import pandas as pd

def plot_gantt(df):
    """
    Dibuja un diagrama de Gantt con colores por job.
    Requiere columnas: job, machine, start, end (y opcionalmente op)
    """
    # Definir colores por job (se generan automáticamente si hay más jobs)
    unique_jobs = sorted(df["job"].unique())
    cmap = plt.get_cmap("tab10")
    job_colors = {job: cmap(i % 10) for i, job in enumerate(unique_jobs)}

    # Configurar figura
    fig, ax = plt.subplots(figsize=(10, 4))
    machines = sorted(df["machine"].unique())[::-1]
    machine_to_y = {m: i for i, m in enumerate(machines)}
    height = 0.6

    # Dibujar barras
    for row in df.itertuples():
        y = machine_to_y[row.machine]
        dur = row.end - row.start
        color = job_colors[row.job]
        ax.barh(y, dur, left=row.start, height=height, color=color)
        label = f"{row.job}-O{row.op}" if hasattr(row, "op") else row.job
        ax.text(row.start + dur / 2, y, label, va="center", ha="center", color="white", fontsize=9)

    # Estética
    ax.set_yticks(list(machine_to_y.values()))
    ax.set_yticklabels(list(machine_to_y.keys()))
    ax.set_xlabel("Time")
    ax.set_title("Gantt Diagram - Makespan: 11")
    ax.grid(axis="x", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig("./unit-04/output/simple_gantt.png")

# ====== Ejemplo de uso (tú defines el orden manualmente) ======
# df = pd.DataFrame([
#     {"job": "J1", "machine": "M1", "start": 0, "end": 3, "op": 1},
#     {"job": "J1", "machine": "M2", "start": 3, "end": 5, "op": 2},
#     {"job": "J2", "machine": "M2", "start": 0, "end": 2, "op": 1},
#     {"job": "J2", "machine": "M1", "start": 2, "end": 3, "op": 2},
# ])
# plot_gantt(df)


# Datos del problema (orden y duración)
import pandas as pd

data = {
    "J1": [("M1", 3), ("M2", 2), ("M3", 2)],
    "J2": [("M2", 2), ("M1", 1), ("M3", 4)],
    "J3": [("M3", 3), ("M2", 1), ("M1", 2)],
}

# Diccionario para llevar el último tiempo disponible por máquina
df = pd.DataFrame([
    {"job": "J1", "machine": "M1", "start": 0, "end": 3, "op": 1},
    {"job": "J1", "machine": "M2", "start": 3, "end": 5, "op": 2},
    {"job": "J1", "machine": "M3", "start": 5, "end": 7, "op": 3},
    {"job": "J2", "machine": "M2", "start": 0, "end": 2, "op": 1},
    {"job": "J2", "machine": "M1", "start": 3, "end": 4, "op": 2},
    {"job": "J2", "machine": "M3", "start": 7, "end": 11, "op": 3},
    {"job": "J3", "machine": "M3", "start": 0, "end": 3, "op": 1},
    {"job": "J3", "machine": "M2", "start": 5, "end": 6, "op": 2},
    {"job": "J3", "machine": "M1", "start": 7, "end": 9, "op": 3},
])
print(df)
plot_gantt(df)