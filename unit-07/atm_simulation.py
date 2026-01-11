import random
import simpy
import statistics
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt


class BankSimulation:
    def __init__(self, seed=None):
        self.seed = seed if seed else random.randint(1, 10000)
        random.seed(self.seed)
        
        self.NEW_CUSTOMERS = 70
        self.INTERVAL_CUSTOMERS = 10.0
        self.MIN_PATIENCE = 1
        self.MAX_PATIENCE = 5
        self.SERVICE_TIME_MEAN = 12.0
        
        self.stats = {
            "total_customers": 0,
            "served_customers": 0,
            "reneged_customers": 0,
            "wait_times_served": [],
            "wait_times_all": [],
            "service_times": [],
            "system_times": [],
            "queue_lengths": [], 
            "server_utilization": 0
        }

        self.queue_length_history = []
        self.server_busy_history = []
        self.current_queue_length = 0
        self.last_event_time = 0
        
    def source(self, env, counter):
        """Generate random customers"""
        for i in range(self.NEW_CUSTOMERS):
            self.stats["total_customers"] += 1
            c = self.customer(env, f"Customer{i:03d}", counter)
            env.process(c)
            
            self.record_queue_state(env)
            
            t = random.expovariate(1.0 / self.INTERVAL_CUSTOMERS)
            yield env.timeout(t)
    
    def customer(self, env, name, counter):
        """Process an individual customer"""
        arrive_time = env.now
        
        self.current_queue_length += 1
        self.record_queue_state(env)
        with counter.request() as req:
            patience = random.uniform(self.MIN_PATIENCE, self.MAX_PATIENCE)
            results = yield req | env.timeout(patience)
            wait_time = env.now - arrive_time

            self.stats["wait_times_all"].append(wait_time)
            
            if req in results:
                self.stats["served_customers"] += 1
                self.stats["wait_times_served"].append(wait_time)
                self.current_queue_length -= 1
                self.record_queue_state(env)
                
                service_time = random.expovariate(1.0 / self.SERVICE_TIME_MEAN)
                self.stats["service_times"].append(service_time)
                self.record_server_utilization(env, service_time, True)
                yield env.timeout(service_time)
                
                system_time = env.now - arrive_time
                self.stats["system_times"].append(system_time)
                self.record_server_utilization(env, 0, False)
            else:
                self.stats["reneged_customers"] += 1
                self.current_queue_length -= 1
                self.record_queue_state(env)
    
    def record_queue_state(self, env):
        self.queue_length_history.append({
            "time": env.now,
            "queue_length": self.current_queue_length
        })
    
    def record_server_utilization(self, env, duration, is_start):
        if is_start:
            self.server_busy_history.append({
                "time": env.now,
                "state": "busy"
            })
        else:
            self.server_busy_history.append({
                "time": env.now,
                "state": "idle"
            })
    
    def run_single_replication(self):
        """Execute one simulation replica"""
        self.__init__(self.seed)
        env = simpy.Environment()
        counter = simpy.Resource(env, capacity=1)
        env.process(self.source(env, counter))
        env.run()
        self.calculate_final_metrics()
        return self.get_summary_stats()
    
    def calculate_final_metrics(self):
        """Calcula Metrics finales basadas en los datos recolectados"""
        # Utilización del servidor (aproximada)
        if self.queue_length_history:
            total_time = self.queue_length_history[-1]["time"]
            busy_time = sum(1 for h in self.server_busy_history if h["state"] == "busy")
            self.stats["server_utilization"] = busy_time / total_time if total_time > 0 else 0
    
    def get_summary_stats(self):
        return {
            "seed": self.seed,
            "total_customers": self.stats["total_customers"],
            "served_customers": self.stats["served_customers"],
            "reneged_customers": self.stats["reneged_customers"],
            "reneged_percentage": (self.stats["reneged_customers"] / self.stats["total_customers"] * 100) 
                                   if self.stats["total_customers"] > 0 else 0,
            "avg_wait_served": statistics.mean(self.stats["wait_times_served"]) 
                               if self.stats["wait_times_served"] else 0,
            "avg_wait_all": statistics.mean(self.stats["wait_times_all"]) 
                            if self.stats["wait_times_all"] else 0,
            "avg_service_time": statistics.mean(self.stats["service_times"]) 
                                if self.stats["service_times"] else 0,
            "avg_system_time": statistics.mean(self.stats["system_times"]) 
                               if self.stats["system_times"] else 0,
            "server_utilization": self.stats["server_utilization"],
            "max_queue_length": max([h["queue_length"] for h in self.queue_length_history]) 
                                if self.queue_length_history else 0,
            "avg_queue_length": statistics.mean([h["queue_length"] for h in self.queue_length_history]) 
                                if self.queue_length_history else 0
        }

# ==============================
# EJECUCIÓN MÚLTIPLE Y ANÁLISIS ESTADÍSTICO
# ==============================

def run_multiple_replications(num_replications=10):
    all_results = []
    
    for _ in tqdm(range(num_replications), desc = "ATM Simulation"):
        sim = BankSimulation()
        results = sim.run_single_replication()
        all_results.append(results)
    
    df_results = pd.DataFrame(all_results)

    summary_stats = {
        "Metric": [],
        "Average": [],
        "Standard Deviation": [],
        "Minimum": [],
        "Maximum": [],
        "IC 95% Lower": [],
        "IC 95% Upper": []
    }
    
    # Metrics a analizar
    metrics_to_analyze = [
        "served_customers",
        "reneged_customers", 
        "reneged_percentage",
        "avg_wait_all",
        "avg_service_time",
        "server_utilization",
        "avg_queue_length"
    ]
    
    for metric in metrics_to_analyze:
        if metric in df_results.columns:
            values = df_results[metric]
            mean_val = np.mean(values)
            std_val = np.std(values)

            n = len(values)
            t_value = stats.t.ppf(0.975, df=n-1)
            ci_low = mean_val - t_value * (std_val / np.sqrt(n))
            ci_high = mean_val + t_value * (std_val / np.sqrt(n))
            
            summary_stats["Metric"].append(metric)
            summary_stats["Average"].append(mean_val)
            summary_stats["Standard Deviation"].append(std_val)
            summary_stats["Minimum"].append(np.min(values))
            summary_stats["Maximum"].append(np.max(values))
            summary_stats["IC 95% Lower"].append(ci_low)
            summary_stats["IC 95% Upper"].append(ci_high)
    
    df_summary = pd.DataFrame(summary_stats)
    return df_results, df_summary


def main():
    N_REPLICATIONS = 1000
    df_results, df_summary = run_multiple_replications(N_REPLICATIONS)
    
    print("\n\nStatistical Summary")
    print(f"{"="*60}")
    print(df_summary.to_string(index=False))

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    # Gráfico de porcentaje de abandono
    avg_reneged = df_summary[df_summary["Metric"] == "reneged_percentage"]["Average"].values[0]
    axs[0].hist(df_results["reneged_percentage"], bins = 50, edgecolor = "black", alpha = 0.7)
    axs[0].axvline(avg_reneged, color = "red", linestyle = "--", label = f"Average ({avg_reneged:.2f})")
    axs[0].set_xlabel("Reneged Percentage")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Distribution of Reneged Percentage")
    axs[0].legend()
    axs[0].grid(True, alpha = 0.3)    
    # Gráfico de tiempo de espera Average
    avg_waiting = df_summary[df_summary["Metric"] == "avg_wait_all"]["Average"].values[0]
    axs[1].hist(df_results["avg_wait_all"], bins = 50, edgecolor = "black", alpha = 0.7, color = "green")
    axs[1].axvline(avg_waiting, color = "red", linestyle = "--", label = f"Average ({avg_waiting:.2f})")
    axs[1].set_xlabel("Average Waiting Time")
    axs[1].set_title("Distribution of Average Waiting Time")
    axs[1].legend()
    axs[1].grid(True, alpha = 0.3)    
    # Gráfico de utilización del servidor
    avg_server = df_summary[df_summary["Metric"] == "server_utilization"]["Average"].values[0]
    axs[2].hist(df_results["server_utilization"], bins = 50, edgecolor = "black", alpha = 0.7, color = "orange")
    axs[2].axvline(avg_server, color = "red", linestyle = "--", label = f"Average ({avg_server:.2f})")
    axs[2].set_xlabel("Server Usage")
    axs[2].set_title("Distribution of Server Usage")
    axs[2].legend()
    axs[2].grid(True, alpha = 0.3)    
    # Gráfico de correlación entre utilización y abandono
    axs[3].scatter(df_results["server_utilization"], df_results["reneged_percentage"], alpha = 0.6)
    axs[3].set_xlabel("Server Usage")
    axs[3].set_ylabel("Reneged Percentage")
    axs[3].set_title("Correlation: Server Usage vs Reneged Percentage")
    axs[3].grid(True, alpha = 0.3)    
    # Calcular y mostrar correlación
    correlation = np.corrcoef(df_results["server_utilization"], df_results["reneged_percentage"])[0, 1]
    axs[3].text(0.05, 0.95, f"Correlation: {correlation:.3f}", transform = axs[3].transAxes, fontsize = 10,
                verticalalignment = "top", bbox = dict(boxstyle = "round", facecolor = "wheat", alpha = 0.5))
    
    plt.savefig("./unit-07/output/atm_simulation.jpg", bbox_inches = "tight")
    

if __name__ == "__main__":
    main()