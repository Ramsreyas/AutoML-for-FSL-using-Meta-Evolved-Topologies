# main.py (Modular Version - Search Only)
import os
import torch
import numpy as np
import random
import time
import json
from deap import tools
from src.data_loader import create_task_generator   
from src.evolution import EvolutionaryEngine
from src.model_builder import build_model_from_genotype
from src.fitness import evaluate_fitness

# --- CONFIGURATION -----------------------------------------------------------
# Select Experiment ID manually here:
# 1 = 5-Way 1-Shot
# 2 = 5-Way 5-Shot
# 3 = 20-Way 1-Shot
EXPERIMENT_ID = 1

if EXPERIMENT_ID == 1:
    EXP_NAME = "5Way_1Shot"
    N_WAY = 5; K_SHOT = 1
elif EXPERIMENT_ID == 2:
    EXP_NAME = "5Way_5Shot"
    N_WAY = 5; K_SHOT = 5
elif EXPERIMENT_ID == 3:
    EXP_NAME = "20Way_1Shot"
    N_WAY = 20; K_SHOT = 1

print(f"--- STARTING SEARCH MODULE: {EXP_NAME} ---")

# Research Constants
SEED = 42
QUERY_SAMPLES = 15
ADAPTATION_STEPS = 5
INNER_LR = 0.1
NUM_TASKS_SEARCH = 50  # Fast proxy
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.6

# Paths
DATASET_ROOT = '/content/data'
DRIVE_ROOT = '/content/drive/MyDrive/AutoML-for-FSL-using-Meta-Evolved-Topologies'
CHECKPOINT_PATH = os.path.join(DRIVE_ROOT, f'checkpoint_{EXP_NAME}.pkl')
RESULTS_PATH = os.path.join(DRIVE_ROOT, f'results_{EXP_NAME}.txt')
CHAMPION_PATH = os.path.join(DRIVE_ROOT, f'champion_{EXP_NAME}.json')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

def save_champion(ind, filepath):
    with open(filepath, 'w') as f:
        json.dump(ind, f)
    print(f"Champion genotype saved to {filepath}")

def main():
    set_seed(SEED)
    print(f"Loading Data ({N_WAY}-Way, {K_SHOT}-Shot)...")
    task_generator = create_task_generator(DATASET_ROOT, N_WAY, K_SHOT, QUERY_SAMPLES)
    
    engine = EvolutionaryEngine(POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE)
    population, start_gen = engine.load_checkpoint(CHECKPOINT_PATH)
    
    if population is None:
        print("Initializing new population...")
        population = engine.initialize_population()
        start_gen = 0
    else:
        print(f"Resuming from Generation {start_gen}...")

    # Evolutionary Loop
    for gen in range(start_gen, GENERATIONS):
        print(f"\n=== GENERATION {gen + 1} / {GENERATIONS} ===")
        start_time = time.time()
        
        # Evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        print(f"Evaluating {len(invalid_ind)} candidates...")
        
        for i, ind in enumerate(invalid_ind):
            try:
                model = build_model_from_genotype(ind, N_WAY)
                score = evaluate_fitness(model, task_generator, N_WAY, K_SHOT, QUERY_SAMPLES, ADAPTATION_STEPS, INNER_LR, NUM_TASKS_SEARCH)
            except:
                score = 0.0
            ind.fitness.values = (score,)
            if (i+1)%5==0: print(f"  Processed {i+1}/{len(invalid_ind)}")

        # Log Stats
        fits = [ind.fitness.values[0] for ind in population]
        best_fit = max(fits)
        avg_fit = sum(fits) / len(population)
        print(f"  Stats -> Max: {best_fit:.4f} | Avg: {avg_fit:.4f}")
        
        with open(RESULTS_PATH, "a") as f:
            f.write(f"Gen {gen+1}: Max={best_fit:.4f}, Avg={avg_fit:.4f}\n")

        # Next Gen (unless last)
        if gen < GENERATIONS - 1:
            population = engine.evolve_next_generation(population)
            engine.save_checkpoint(population, gen + 1, CHECKPOINT_PATH)
        
        print(f"Gen took {time.time() - start_time:.2f}s")

    # Save Champion
    print("\n--- Search Complete. Saving Champion... ---")
    best_ind = tools.selBest(population, 1)[0]
    save_champion(best_ind, CHAMPION_PATH)
    print(f"Best Search Score: {best_ind.fitness.values[0]:.4f}")

if __name__ == "__main__":
    main()