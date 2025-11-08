import time
import random
import operator
import matplotlib.pyplot as plt
import numpy as np

class Job:
    def __init__(self, ID, processing_times):
        self.ID = ID
        self.processing_times = processing_times  # List of processing times for each machine
        self.TPT = sum(processing_times)  # Total processing time

class Solution:
    last_ID = -1
    
    def __init__(self, n_jobs, n_machines):
        Solution.last_ID += 1
        self.ID = Solution.last_ID
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.jobs = []  # List of jobs in sequence
        self.makespan = 0.0

    def calc_makespan(self):
        """
        Calculate makespan using traditional method (for verification)
        """
        nRows = self.n_jobs
        nCols = self.n_machines
        times = [[0 for _ in range(nCols)] for _ in range(nRows)]
        
        for i in range(nRows):
            for j in range(nCols):
                if i == 0 and j == 0:
                    times[0][0] = self.jobs[0].processing_times[0]
                elif i == 0:
                    times[0][j] = times[0][j-1] + self.jobs[0].processing_times[j]
                elif j == 0:
                    times[i][0] = times[i-1][0] + self.jobs[i].processing_times[0]
                else:
                    max_time = max(times[i-1][j], times[i][j-1])
                    times[i][j] = max_time + self.jobs[i].processing_times[j]
        
        self.makespan = times[nRows-1][nCols-1]
        return self.makespan

def read_pfsp_data(instance_name):
    """
    Read PFSP instance data from file
    Format based on Taillard benchmarks
    """
    with open(f"./data/pfsp/{instance_name}_inputs.txt", "r") as file:
        lines = file.readlines()
    n_jobs, n_machines = map(int, lines[1].split())
    jobs = []
    job_id = 0
    for i in range(3, 3 + n_jobs):
        processing_times = list(map(float, lines[i].split()))
        job = Job(job_id, processing_times)
        jobs.append(job)
        job_id += 1
        
    return jobs, n_jobs, n_machines

def generate_shortest_processing_time_first(jobs, n_jobs, n_machines):
    """
    Generate SPT solution (shortest processing time first on first machine)
    """    
    solution = Solution(n_jobs, n_machines)
    # Sort by processing time on first machine
    sorted_jobs = sorted(jobs, key=lambda x: x.processing_times[0])
    solution.jobs = sorted_jobs
    solution.calc_makespan()
    return solution

def calc_e_matrix(solution, k):
    """
    Calculate E matrix for Taillard's acceleration
    E[i][j] = earliest completion time of job i on machine j
    """
    n_machines = solution.n_machines
    e = [[0.0 for _ in range(n_machines)] for _ in range(k+1)]
    
    for i in range(k+1):
        for j in range(n_machines):
            if i == 0 and j == 0:
                e[i][j] = solution.jobs[i].processing_times[j]
            elif i == 0:
                e[i][j] = e[i][j-1] + solution.jobs[i].processing_times[j]
            elif j == 0:
                e[i][j] = e[i-1][j] + solution.jobs[i].processing_times[j]
            else:
                max_time = max(e[i-1][j], e[i][j-1])
                e[i][j] = max_time + solution.jobs[i].processing_times[j]
    
    return e

def calc_q_matrix(solution, k):
    """
    Calculate Q matrix for Taillard's acceleration
    Q[i][j] = tail time from job i on machine j to the end
    """
    n_machines = solution.n_machines
    q = [[0.0 for _ in range(n_machines)] for _ in range(k+1)]
    
    for i in range(k, -1, -1):
        for j in range(n_machines-1, -1, -1):
            if i == k and j == n_machines-1:
                q[i][j] = solution.jobs[i].processing_times[j]
            elif i == k:
                q[i][j] = q[i][j+1] + solution.jobs[i].processing_times[j]
            elif j == n_machines-1:
                q[i][j] = q[i+1][j] + solution.jobs[i].processing_times[j]
            else:
                max_time = max(q[i+1][j], q[i][j+1])
                q[i][j] = max_time + solution.jobs[i].processing_times[j]
    
    return q

def calc_f_matrix(solution, k, e_matrix, inserted_job):
    """
    Calculate F matrix for Taillard's acceleration
    F[i][j] = earliest completion time if inserted_job is placed at position i
    """
    n_machines = solution.n_machines
    f = [[0.0 for _ in range(n_machines)] for _ in range(k+1)]
    
    for i in range(k+1):
        for j in range(n_machines):
            if i == 0 and j == 0:
                f[i][j] = inserted_job.processing_times[j]
            elif i == 0:
                f[i][j] = f[i][j-1] + inserted_job.processing_times[j]
            elif j == 0:
                f[i][j] = e_matrix[i-1][j] + inserted_job.processing_times[j]
            else:
                max_time = max(e_matrix[i-1][j], f[i][j-1])
                f[i][j] = max_time + inserted_job.processing_times[j]
    
    return f

def improve_by_shifting_job_to_left(solution, k):
    """
    Find the best position for the job at position k by shifting it left
    Uses Taillard's acceleration for fast makespan computation
    """
    if k == 0:
        return solution  # First job cannot be shifted
    
    best_position = k
    min_makespan = float("inf")
    
    # Get the job to be inserted
    inserted_job = solution.jobs[k]
    
    # Calculate matrices for acceleration
    e_matrix = calc_e_matrix(solution, k-1)  # E matrix without the last job
    q_matrix = calc_q_matrix(solution, k)
    f_matrix = calc_f_matrix(solution, k, e_matrix, inserted_job)
    
    # Evaluate all possible positions
    for i in range(k, -1, -1):
        max_sum = 0.0
        for j in range(solution.n_machines):
            new_sum = f_matrix[i][j] + q_matrix[i][j]
            if new_sum > max_sum:
                max_sum = new_sum
        
        new_makespan = max_sum
        
        # In case of tie, prefer leftmost position
        if new_makespan <= min_makespan:
            min_makespan = new_makespan
            best_position = i
    
    # Update solution if improvement found
    if best_position < k:
        # Remove job from current position
        aux_job = solution.jobs.pop(k)
        # Insert at best position
        solution.jobs.insert(best_position, aux_job)
        
        # Update makespan if this is the final job
        if k == solution.n_jobs - 1:
            solution.makespan = min_makespan
    
    return solution

def neh_heuristic(jobs, n_jobs, n_machines):
    """
    NEH Heuristic for Permutation Flow-Shop Problem
    """
    start_time = time.time()
    sorted_jobs = sorted(jobs, key=operator.attrgetter("TPT"), reverse=True)
    solution = Solution(n_jobs, n_machines)
    solution.jobs.append(sorted_jobs[0])
    
    for i in range(1, n_jobs):
        solution.jobs.append(sorted_jobs[i])
        solution = improve_by_shifting_job_to_left(solution, i)

    solution.calc_makespan()
    end_time = time.time()
    computational_time = end_time - start_time
    
    return solution, computational_time

def plot_gantt_chart(solution, instance_name):
    """
    Plot Gantt chart for the PFSP solution
    """
    n_jobs = solution.n_jobs
    n_machines = solution.n_machines
    
    # Calculate start and end times for each operation
    start_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    end_times = [[0 for _ in range(n_machines)] for _ in range(n_jobs)]
    
    # Compute schedule
    for i in range(n_jobs):
        for j in range(n_machines):
            if i == 0 and j == 0:
                start_times[i][j] = 0
            elif i == 0:
                start_times[i][j] = end_times[i][j-1]
            elif j == 0:
                start_times[i][j] = end_times[i-1][j]
            else:
                start_times[i][j] = max(end_times[i-1][j], end_times[i][j-1])
            
            end_times[i][j] = start_times[i][j] + solution.jobs[i].processing_times[j]
    
    # Create Gantt chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_jobs))
    
    for i in range(n_jobs):
        for j in range(n_machines):
            start = start_times[i][j]
            duration = solution.jobs[i].processing_times[j]
            ax.barh(j, duration, left=start, height=0.6, 
                   color=colors[i], edgecolor="black", alpha=0.7, label = f"Job {i}" if j==0 else None)
            
            # Add job ID label
            ax.text(start + duration/2, j, f"J{solution.jobs[i].ID}", 
                   ha="center", va="center", fontweight="bold", fontsize=8)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{m+1}" for m in range(n_machines)])
    ax.set_title(f"PFSP Gantt Chart - {instance_name}\nMakespan: {solution.makespan:.2f}")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"./unit-05/output/{instance_name}_gantt.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    instance_name = "tai120_500_20"
    jobs, n_jobs, n_machines = read_pfsp_data(instance_name)
    
    # Run NEH heuristic
    baseline_solution = generate_shortest_processing_time_first(jobs, n_jobs, n_machines)
    best_solution, computational_time = neh_heuristic(jobs, n_jobs, n_machines)
    
    print(f"Instance: {instance_name}")
    print(f"Number of jobs: {best_solution.n_jobs}")
    print(f"Number of machines: {best_solution.n_machines}")

    print(f"Baseline Makespan (shortest processing time first on first machine): {baseline_solution.makespan:.2f}")
    print(f"NEH Makespan: {best_solution.makespan:.2f}")
    print(f"Computational time: {computational_time:.4f} seconds")
    # Create visualizations
    plot_gantt_chart(best_solution, instance_name)