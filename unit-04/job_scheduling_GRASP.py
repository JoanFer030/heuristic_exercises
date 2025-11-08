# Instances: https://github.com/thomasWeise/jsspInstancesAndResults
import copy
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class JSSPInstance:
    def __init__(self):
        self.n_jobs = None
        self.n_machines = None
        self.processing_times = []
        self.machine_sequence = []
        self.best_known_makespan = None
    
    def load_instance(self, instance_name: str):
        with open("./data/jssp/equivalences.json", "r") as file:
            equivalences = json.loads(file.read())
        if instance_name in equivalences:
            self.best_known_makespan = equivalences[instance_name]["bks"]
        else:
            raise ValueError("Instance not available")
        
        with open(f"./data/jssp/{instance_name}.txt", "r") as file:
            first_line = file.readline().strip().split()
            self.n_jobs = int(first_line[0])
            self.n_machines = int(first_line[1])
            for _ in range(self.n_jobs):
                line = file.readline().strip().split()
                if not line:
                    continue
                data = [int(x) for x in line]

                machines = []
                times = []
                for i in range(0, len(data), 2):
                    if i + 1 < len(data):
                        machines.append(data[i])
                        times.append(data[i + 1])
                self.machine_sequence.append(machines)
                self.processing_times.append(times)


class JSSPSolution:
    def __init__(self, instance: JSSPInstance):
        self.instance = instance
        self.schedule = {}  # {(job, operation): start_time}
        self.makespan = 0
        self.machine_schedules = {m: [] for m in range(instance.n_machines)}
        
    def calculate_makespan(self):
        if not self.schedule:
            return 0
            
        end_times = []
        for (job, op), start_time in self.schedule.items():
            duration = self.instance.processing_times[job][op]
            end_times.append(start_time + duration)
        
        self.makespan = max(end_times) if end_times else 0
        return self.makespan
    
    def is_feasible(self):
        # All operations are scheduled
        scheduled_ops = set(self.schedule.keys())
        required_ops = {(job, op) for job in range(self.instance.n_jobs) 
                       for op in range(self.instance.n_machines)}
        if scheduled_ops != required_ops:
            return False
            
        for job in range(self.instance.n_jobs):
            op_times = []
            for op in range(self.instance.n_machines):
                if (job, op) in self.schedule:
                    start_time = self.schedule[(job, op)]
                    duration = self.instance.processing_times[job][op]
                    op_times.append((start_time, start_time + duration))
                else:
                    return False
            # No overlap
            for i in range(1, len(op_times)):
                if op_times[i][0] < op_times[i-1][1]:
                    return False
        return True
    

def greedy_solution(instance: JSSPInstance) -> JSSPSolution:
    """
    Solución greedy: siempre selecciona la operación disponible con menor tiempo de procesamiento
    """
    solution = JSSPSolution(instance)
    machine_available_time = [0] * instance.n_machines
    job_next_operation = [0] * instance.n_jobs
    job_available_time = [0] * instance.n_jobs
    
    available_operations = []
    for job in range(instance.n_jobs):
        if job_next_operation[job] < instance.n_machines:
            op = job_next_operation[job]
            machine = instance.machine_sequence[job][op]
            processing_time = instance.processing_times[job][op]
            available_operations.append((job, op, machine, processing_time))
    
    while available_operations:
        available_operations.sort(key=lambda x: x[3])
        job, op, machine, processing_time = available_operations.pop(0)
        start_time = max(machine_available_time[machine], job_available_time[job])
        
        solution.schedule[(job, op)] = start_time
        end_time = start_time + processing_time
        
        machine_available_time[machine] = end_time
        job_available_time[job] = end_time
        job_next_operation[job] += 1
        
        if job_next_operation[job] < instance.n_machines:
            next_op = job_next_operation[job]
            next_machine = instance.machine_sequence[job][next_op]
            next_processing_time = instance.processing_times[job][next_op]
            available_operations.append((job, next_op, next_machine, next_processing_time))
    solution.makespan = solution.calculate_makespan()
    return solution

def grasp_solution(instance: JSSPInstance, alpha: float, max_iterations: int = 50, 
                  rcl_size: int = 3, local_search_iterations: int = 20) -> tuple[JSSPSolution, list]:
    best_solution = None
    best_makespan = float("inf")
    ms_history = []
    
    for iteration in range(max_iterations):
        solution = grasp_construction(instance, rcl_size, alpha)
        if solution is None:
            continue
        improved_solution = local_search(instance, solution, local_search_iterations)
        if improved_solution.makespan < best_makespan:
            best_solution = improved_solution
            best_makespan = improved_solution.makespan
            ms_history.append((iteration, best_makespan))
    
    return best_solution, ms_history

def grasp_construction(instance: JSSPInstance, rcl_size: int, alpha: float) -> JSSPSolution:
    solution = JSSPSolution(instance)
    
    machine_available_time = [0] * instance.n_machines
    job_next_operation = [0] * instance.n_jobs
    job_available_time = [0] * instance.n_jobs
    
    available_operations = []
    for job in range(instance.n_jobs):
        if job_next_operation[job] < instance.n_machines:
            op = job_next_operation[job]
            machine = instance.machine_sequence[job][op]
            processing_time = instance.processing_times[job][op]
            available_operations.append((job, op, machine, processing_time))
    
    while available_operations:
        candidate_operations = []
        for job, op, machine, processing_time in available_operations:
            start_time = max(machine_available_time[machine], job_available_time[job])
            end_time = start_time + processing_time
            candidate_operations.append((job, op, machine, processing_time, end_time))
        
        if not candidate_operations:
            break
            
        # Create Restricted Candidate List (RCL) 
        candidate_operations.sort(key=lambda x: x[4])
        best_end_time = candidate_operations[0][4]
        worst_end_time = candidate_operations[-1][4]
        # Calcular threshold para RCL
        threshold = best_end_time + alpha * (worst_end_time - best_end_time)
        rcl = [op for op in candidate_operations if op[4] <= threshold]
        if len(rcl) > rcl_size:
            rcl = rcl[:rcl_size]
        elif not rcl:
            rcl = candidate_operations[:min(rcl_size, len(candidate_operations))]
        
        # Seleccionar aleatoriamente de la RCL
        selected = random.choice(rcl)
        job, op, machine, processing_time, _ = selected
        available_operations = [op_data for op_data in available_operations 
                              if not (op_data[0] == job and op_data[1] == op)]
        
        # Programar la operación
        start_time = max(machine_available_time[machine], job_available_time[job])
        solution.schedule[(job, op)] = start_time
        end_time = start_time + processing_time
        
        # Actualizar tiempos
        machine_available_time[machine] = end_time
        job_available_time[job] = end_time
        job_next_operation[job] += 1
        
        if job_next_operation[job] < instance.n_machines:
            next_op = job_next_operation[job]
            next_machine = instance.machine_sequence[job][next_op]
            next_processing_time = instance.processing_times[job][next_op]
            available_operations.append((job, next_op, next_machine, next_processing_time))
    solution.makespan = solution.calculate_makespan()
    return solution

def local_search(instance: JSSPInstance, solution: JSSPSolution, 
                iterations: int) -> JSSPSolution:
    current_solution = copy.deepcopy(solution)
    best_solution = current_solution
    best_makespan = current_solution.makespan
    
    for _ in range(iterations):
        machine_schedules = {m: [] for m in range(instance.n_machines)}
        for (job, op), start_time in current_solution.schedule.items():
            machine = instance.machine_sequence[job][op]
            duration = instance.processing_times[job][op]
            machine_schedules[machine].append((job, op, start_time, duration))
        for machine in machine_schedules:
            machine_schedules[machine].sort(key=lambda x: x[2])
        machine_end_times = []
        for machine, schedules in machine_schedules.items():
            if schedules:
                last_op = schedules[-1]
                end_time = last_op[2] + last_op[3]
                machine_end_times.append((machine, end_time))
        if not machine_end_times:
            continue

        machine_end_times.sort(key=lambda x: x[1], reverse=True)
        critical_machine = machine_end_times[0][0]
        operations = machine_schedules[critical_machine]
        if len(operations) <= 1:
            continue
            

        new_solution = copy.deepcopy(current_solution)
        idx = random.randint(0, len(operations) - 2)
        op1 = operations[idx]
        op2 = operations[idx + 1]
        job1, op_idx1, start1, dur1 = op1
        job2, op_idx2, start2, dur2 = op2
        new_start1 = start2
        new_start2 = start1
        new_solution.schedule[(job1, op_idx1)] = new_start1
        new_solution.schedule[(job2, op_idx2)] = new_start2
        
        # Recalcular makespan
        new_solution.makespan = new_solution.calculate_makespan()
        if new_solution.makespan <= best_makespan:
            current_solution = new_solution
            if new_solution.makespan < best_makespan:
                best_solution = new_solution
                best_makespan = new_solution.makespan
    return best_solution

def plot_gantt_chart(solution: JSSPSolution,instance_name: str):
    """
    Genera un diagrama de Gantt para visualizar la solución
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    machine_schedules = {m: [] for m in range(solution.instance.n_machines)}
    for (job, op), start_time in solution.schedule.items():
        machine = solution.instance.machine_sequence[job][op]
        duration = solution.instance.processing_times[job][op]
        machine_schedules[machine].append((job, op, start_time, duration))
    
    colors = plt.cm.Set3(np.linspace(0, 1, solution.instance.n_jobs))
    for machine in range(solution.instance.n_machines):
        for job, op, start, duration in machine_schedules[machine]:
            rect = patches.Rectangle((start, machine - 0.4), duration, 0.8, linewidth = 1, edgecolor = "black", facecolor = colors[job], alpha = 0.7)
            ax.add_patch(rect)
            ax.text(start + duration/2, machine, f"O{op}", ha = "center", va = "center", fontweight = "bold", fontsize = 8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title(f"GRASP Solution for {instance_name}\nMakespan: {solution.makespan}")
    ax.set_yticks(range(solution.instance.n_machines))
    ax.set_yticklabels([f"Machine {i+1}" for i in range(solution.instance.n_machines)])
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, solution.makespan * 1.1)
    ax.set_ylim(-0.5, solution.instance.n_machines - 0.5)
    legend_patches = [patches.Patch(color=colors[i], label=f"Job {i+1}") for i in range(solution.instance.n_jobs)]
    ax.legend(handles=legend_patches, loc="upper right")
    plt.tight_layout()
    plt.savefig(f"./unit-04/output/{instance_name}_GRASP.png")

def plot_convergence(ms_history, greedy_ms, bk_ms, instance_name):
    iterations, costs = zip(*ms_history)
    plt.figure(figsize = (10, 6))

    plt.step(iterations, costs, "b-", linewidth=2, label = f"GRASP: {min(costs):.2f}")
    plt.axhline(y = greedy_ms, color = "r", linestyle = "--", linewidth = 2, label = f"Greedy Algorithm: {greedy_ms}")
    plt.axhline(y = bk_ms, color = "g", linestyle = "--", linewidth = 2, label = f"Best Known Solution: {bk_ms}")
    plt.xlabel("Iteration")
    plt.ylabel("Makespan")
    plt.title(f"Job-Shop Scheduling Problem - {instance_name}")
    plt.xscale("log")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"./unit-04/output/{instance_name}_convergence_GRASP.png")

#######################################################################
###                         SOLVING & RESULTS                       ###
#######################################################################
if __name__ == "__main__":
    instance_name = "cscmax_20_15_1"
    alpha = 0.3
    max_iterations = 500
    ls_iterations = 50
    rcl_size = 3

    instance = JSSPInstance()
    instance.load_instance(instance_name)
    
    # Greedy Solution
    greedy_sol = greedy_solution(instance)
    
    # GRASP Solution
    t0 = time.time()
    grasp_sol, ms_hist = grasp_solution(instance, alpha, max_iterations = max_iterations, 
                                                         rcl_size = rcl_size, 
                                                         local_search_iterations = ls_iterations)
    elapsed_time = time.time() - t0
                                                         
    # Results
    print(f"Instance: {instance_name}")
    print(f"Best known makespan: {instance.best_known_makespan:.2f}")
    print(f"Greedy makespan: {greedy_sol.makespan:.2f}")
    print(f"GRASP makespan: {grasp_sol.makespan:.2f}  |  Computational time: {elapsed_time:.4f}s")

    plot_gantt_chart(grasp_sol, instance_name)
    plot_convergence(ms_hist, greedy_sol.makespan, instance.best_known_makespan, instance_name)
    