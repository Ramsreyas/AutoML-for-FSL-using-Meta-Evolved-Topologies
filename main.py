# main.py
# ==============================================================================
# PROJECT: AutoML for FSL (Master Experiment Runner)
# ==============================================================================

import os
import json
import torch
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from deap import tools

from src.data_loader import create_task_generator
from src.evolution import EvolutionaryEngine
from src.model_builder import build_model_from_genotype, build_baseline_cnn
from src.fitness import evaluate_fitness

# --- CONFIGURATION (Fixed for Research) ---
SEED = 42
DRIVE_ROOT = '/content/drive/MyDrive/AutoML-for-FSL-using-Meta-Evolved-Topologies'
DATA_ROOT = '/content/data'

# Evolution Params
POP_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.6
SEARCH_TASKS = 50       # Fast proxy for evolution

# Validation Params
VALIDATION_TASKS = 1000 # Strict IEEE standard
INNER_LR = 0.1          # Stable SGD learning rate
ADAPTATION_STEPS = 5
QUERY_SAMPLES = 15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

def run_experiment(n_way, k_shot, exp_name):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EXPERIMENT: {exp_name} ({n_way}-Way {k_shot}-Shot)")
    print(f"{'='*60}\n")
    
    # 1. Setup Folders
    exp_dir = os.path.join(DRIVE_ROOT, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    results_path = os.path.join(exp_dir, 'search_log.txt')
    checkpoint_path = os.path.join(exp_dir, 'checkpoint.pkl')
    champion_path = os.path.join(exp_dir, 'champion.json')
    graph_path = os.path.join(exp_dir, 'evolution_graph.png')
    report_path = os.path.join(exp_dir, 'final_report.txt')

    set_seed(SEED)

    # --- PHASE 1: EVOLUTIONARY SEARCH ---
    print("--- [1/4] Evolutionary Search (Finding Champion) ---")
    task_gen_search = create_task_generator(DATA_ROOT, n_way, k_shot, QUERY_SAMPLES)
    engine = EvolutionaryEngine(POP_SIZE, MUTATION_RATE, CROSSOVER_RATE)
    
    # Load or Start Fresh
    population, start_gen = engine.load_checkpoint(checkpoint_path)
    if population is None:
        population = engine.initialize_population()
        start_gen = 0
        if os.path.exists(results_path): os.remove(results_path) # Clean log for new run

    # Evolution Loop
    for gen in range(start_gen, GENERATIONS):
        print(f"  Generation {gen+1}/{GENERATIONS}...")
        start_time = time.time()
        
        # Evaluate Fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        for ind in invalid_ind:
            try:
                model = build_model_from_genotype(ind, n_way)
                score = evaluate_fitness(model, task_gen_search, n_way, k_shot, QUERY_SAMPLES, ADAPTATION_STEPS, INNER_LR, SEARCH_TASKS)
            except: score = 0.0
            ind.fitness.values = (score,)
            
        # Logging
        fits = [ind.fitness.values[0] for ind in population]
        best_fit = max(fits)
        avg_fit = sum(fits) / len(population)
        
        with open(results_path, "a") as f:
            f.write(f"Gen {gen+1}: Max={best_fit:.4f}, Avg={avg_fit:.4f}\n")
            
        # Evolve Next Gen
        if gen < GENERATIONS - 1:
            population = engine.evolve_next_generation(population)
            engine.save_checkpoint(population, gen+1, checkpoint_path)
            
        print(f"  > Max: {best_fit:.4f} | Avg: {avg_fit:.4f} | Time: {time.time() - start_time:.1f}s")
    
    # Save Champion
    best_ind = tools.selBest(population, 1)[0]
    with open(champion_path, 'w') as f:
        json.dump(best_ind, f)
    print(f"✅ Search Complete. Champion DNA saved.")


    # --- PHASE 2: GENERATE GRAPH ---
    print("\n--- [2/4] Generating Evolution Graph ---")
    try:
        gens, maxs, avgs = [], [], []
        with open(results_path, 'r') as f:
            for line in f:
                if "Max=" not in line: continue
                
                parts = line.strip().split()
                # FIX: Added .replace(',', '') to handle trailing commas
                max_val = float(parts[2].split('=')[1].replace(',', ''))
                avg_val = float(parts[3].split('=')[1].replace(',', ''))
                
                maxs.append(max_val)
                avgs.append(avg_val)
                gens.append(len(maxs))
        
                
        plt.figure(figsize=(10,6))
        plt.plot(gens, maxs, 'r-o', linewidth=2, label='Champion Accuracy')
        plt.plot(gens, avgs, 'b--', linewidth=2, label='Population Average')
        plt.title(f'Architecture Evolution: {exp_name}')
        plt.xlabel('Generation'); plt.ylabel('Validation Accuracy')
        plt.grid(True, alpha=0.3); plt.legend()
        plt.savefig(graph_path)
        plt.close()
        print(f"✅ Graph saved to {graph_path}")
    except Exception as e:
        print(f"⚠️ Graph generation failed: {e}")

    # --- PHASE 3: BASELINE COMPARISON ---
    print("\n--- [3/4] Baseline Benchmark (Standard Conv-4) ---")
    print(f"  Running rigorous test on {VALIDATION_TASKS} tasks...")
    # We create a new generator for validation to ensure randomness
    task_gen_val = create_task_generator(DATA_ROOT, n_way, k_shot, QUERY_SAMPLES)
    
    baseline_model = build_baseline_cnn(n_way)
    base_acc = evaluate_fitness(baseline_model, task_gen_val, n_way, k_shot, QUERY_SAMPLES, ADAPTATION_STEPS, INNER_LR, VALIDATION_TASKS)
    print(f"  > Baseline Result: {base_acc:.4f}")

    # --- PHASE 4: CHAMPION VALIDATION ---
    print("\n--- [4/4] Champion Validation (Evolved Model) ---")
    print(f"  Running rigorous test on {VALIDATION_TASKS} tasks...")
    
    champ_model = build_model_from_genotype(best_ind, n_way)
    champ_acc = evaluate_fitness(champ_model, task_gen_val, n_way, k_shot, QUERY_SAMPLES, ADAPTATION_STEPS, INNER_LR, VALIDATION_TASKS)
    print(f"  > Champion Result: {champ_acc:.4f}")

    # --- FINAL REPORT ---
    with open(report_path, 'w') as f:
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Configuration: {n_way}-Way {k_shot}-Shot\n")
        f.write("-" * 30 + "\n")
        f.write(f"Baseline Accuracy (Conv-4): {base_acc:.4f}\n")
        f.write(f"Champion Accuracy (Evolved): {champ_acc:.4f}\n")
        f.write(f"Absolute Improvement: {champ_acc - base_acc:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(f"Champion DNA: {best_ind}\n")
    
    print(f"\n🎉 EXPERIMENT COMPLETE! Report saved to: {report_path}")

if __name__ == "__main__":
    # === UNCOMMENT ONLY ONE LINE BELOW TO RUN AN EXPERIMENT ===
    
    # run_experiment(n_way=5, k_shot=1, exp_name="EXP1_5Way_1Shot")
    
    # run_experiment(n_way=5, k_shot=5, exp_name="EXP2_5Way_5Shot")
    
    # run_experiment(n_way=20, k_shot=1, exp_name="EXP3_20Way_1Shot")
    
    run_experiment(n_way=20, k_shot=5, exp_name="EXP4_20Way_5Shot")