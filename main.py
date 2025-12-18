# main.py
# ==============================================================================
# PROJECT: AutoML for FSL using Meta-Evolved Topologies
# MODULE: Main Execution Script (M5)
# ==============================================================================

import os
import torch
import numpy as np
import random
import time
from src.data_loader import create_task_generator
from src.evolution import EvolutionaryEngine
from src.model_builder import build_model_from_genotype
from src.fitness import evaluate_fitness

# --- CONFIGURATION (The Control Panel) ---
# Research Standard: Use a fixed seed for reproducibility
SEED = 42
N_WAY = 5
K_SHOT = 1
QUERY_SAMPLES = 15
ADAPTATION_STEPS = 5
INNER_LR = 0.5
NUM_TASKS_PER_EVAL = 50  # 50 tasks for fast evolution, we validate the best later with 500

# Evolutionary Parameters
POPULATION_SIZE = 20
GENERATIONS = 10
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.6

# Paths
DATASET_ROOT = '/content/data'
CHECKPOINT_PATH = '/content/drive/MyDrive/AutoML-for-FSL-using-Meta-Evolved-Topologies/experiment_checkpoint.pkl'
RESULTS_PATH = '/content/drive/MyDrive/AutoML-for-FSL-using-Meta-Evolved-Topologies/results.txt'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main():
    print("--- Starting AutoML-FSL Evolutionary Search ---")
    set_seed(SEED)
    
    # 1. Initialize M1: Data Loader
    print(f"Loading Data (N={N_WAY}, K={K_SHOT})...")
    task_generator = create_task_generator(DATASET_ROOT, N_WAY, K_SHOT, QUERY_SAMPLES)
    
    # 2. Initialize M2: Evolutionary Engine
    engine = EvolutionaryEngine(
        population_size=POPULATION_SIZE, 
        mutation_rate=MUTATION_RATE, 
        crossover_rate=CROSSOVER_RATE
    )
    
    # 3. Check for Checkpoint (The Safety Net)
    population, start_gen = engine.load_checkpoint(CHECKPOINT_PATH)
    
    if population is None:
        print("No checkpoint found. Initializing new population...")
        population = engine.initialize_population()
        start_gen = 0
    else:
        print(f"Resuming from Generation {start_gen}...")

    # 4. The Main Evolutionary Loop
    for gen in range(start_gen, GENERATIONS):
        print(f"\n=== GENERATION {gen + 1} / {GENERATIONS} ===")
        start_time = time.time()
        
        # A. Evaluation Step (M4)
        # We only evaluate individuals that have invalid fitness (newly created or mutated)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        print(f"Evaluating {len(invalid_ind)} new/modified architectures...")
        
        for i, ind in enumerate(invalid_ind):
            # M3: Build the Model
            try:
                model = build_model_from_genotype(ind, N_WAY)
                
                # M4: Evaluate Fitness
                fitness_score = evaluate_fitness(
                    model=model,
                    task_generator=task_generator,
                    ways=N_WAY,
                    shots=K_SHOT,
                    query_samples=QUERY_SAMPLES,
                    adaptation_steps=ADAPTATION_STEPS,
                    inner_lr=INNER_LR,
                    num_tasks_to_test=NUM_TASKS_PER_EVAL
                )
            except Exception as e:
                print(f"  [Warning] Invalid architecture generated. Assigning 0 fitness. Error: {e}")
                fitness_score = 0.0

            # Assign fitness tuple (DEAP requires a tuple)
            ind.fitness.values = (fitness_score,)
            
            # Simple progress log
            if (i+1) % 5 == 0:
                print(f"  Processed {i+1}/{len(invalid_ind)} candidates.")

        # B. Logging Statistics
        fits = [ind.fitness.values[0] for ind in population]
        best_fit = max(fits)
        avg_fit = sum(fits) / len(population)
        print(f"  Stats -> Max: {best_fit:.4f} | Avg: {avg_fit:.4f}")
        
        # Save Log to Drive
        with open(RESULTS_PATH, "a") as f:
            f.write(f"Gen {gen+1}: Max={best_fit:.4f}, Avg={avg_fit:.4f}\n")

        # C. Evolution Step (Create Next Generation)
        # We skip this for the very last generation so we can keep the final population
        if gen < GENERATIONS - 1:
            population = engine.evolve_next_generation(population)
            
            # D. Save Checkpoint (Crucial!)
            engine.save_checkpoint(population, gen + 1, CHECKPOINT_PATH)

        print(f"Generation took {time.time() - start_time:.2f} seconds.")

    # 5. Conclusion
    print("\n--- Evolution Complete ---")
    best_ind = tools.selBest(population, 1)[0]
    print(f"Best Genotype Found: {best_ind}")
    print(f"Best Fitness: {best_ind.fitness.values[0]:.4f}")

if __name__ == "__main__":
    main()