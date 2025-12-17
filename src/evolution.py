# src/evolution.py
# ==============================================================================
# MODULE: EVOLUTIONARY ENGINE (M2)
# Research Standard: Uses DEAP for robust genetic algorithm management.
# Includes 'Safety Net' checkpointing to prevent data loss.
# ==============================================================================

import random
import pickle
import os
import copy
from deap import base, creator, tools
from . import evolution_search_space as ess

# --- 1. DEAP Configuration ---
# We define a "Fitness" class. weights=(1.0,) means we want to MAXIMIZE accuracy.
# (If we were minimizing error, we would use -1.0).
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
# We define an "Individual" class. It is a Python list (the genotype)
# but with an added .fitness attribute to store the M4 score.
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)

class EvolutionaryEngine:
    def __init__(self, population_size=20, mutation_rate=0.2, crossover_rate=0.5):
        """
        Initializes the Genetic Algorithm engine.
        Args:
            population_size: Number of models per generation (Standard: 20-50 for Micro-NAS).
            mutation_rate: Probability of a child mutating (Standard: 0.1 - 0.3).
            crossover_rate: Probability of parents mixing DNA (Standard: 0.5 - 0.7).
        """
        self.pop_size = population_size
        self.mut_rate = mutation_rate
        self.cx_rate = crossover_rate
        
        self.toolbox = base.Toolbox()
        self._register_operators()

    def _register_operators(self):
        """
        Registers the core genetic logic with DEAP.
        """
        # 1. Attribute Generator: How to pick a single gene
        self.toolbox.register("attr_gene", random.choice, ess.GENE_POOL)

        # 2. Individual Generator: How to create a random architecture
        self.toolbox.register("individual", self._create_random_individual)
        
        # 3. Population Generator: How to create a whole generation
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # 4. Selection Operator: Tournament Selection
        # Research Standard: Tournament (k=3) preserves diversity better than Roulette Wheel.
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # 5. Crossover Operator: Two-Point Crossover
        # Swaps a segment of DNA between two parents.
        self.toolbox.register("mate", tools.cxTwoPoint)

        # 6. Mutation Operator: Our Custom Logic
        self.toolbox.register("mutate", self._custom_mutation)

    def _create_random_individual(self):
        """Creates a valid random architecture within constraints."""
        # Random depth between MIN and MAX
        size = random.randint(ess.MIN_LAYERS, ess.MAX_LAYERS)
        
        # Keep generating until we find a valid one (rejection sampling)
        while True:
            genotype = [self.toolbox.attr_gene() for _ in range(size)]
            if ess.is_valid_architecture(genotype):
                return creator.Individual(genotype)

    def _custom_mutation(self, individual):
        """
        Custom mutation operator for list-based genotypes.
        Research Standard: Structural Mutation (Add/Remove/Modify layers).
        """
        mutation_type = random.choice(['add', 'remove', 'modify'])
        
        # We clone the individual to ensure we don't modify it in place if the mutation fails
        mutant = copy.deepcopy(individual)
        
        if mutation_type == 'add' and len(mutant) < ess.MAX_LAYERS:
            idx = random.randint(0, len(mutant))
            mutant.insert(idx, self.toolbox.attr_gene())
            
        elif mutation_type == 'remove' and len(mutant) > ess.MIN_LAYERS:
            idx = random.randint(0, len(mutant) - 1)
            del mutant[idx]
            
        elif mutation_type == 'modify':
            idx = random.randint(0, len(mutant) - 1)
            mutant[idx] = self.toolbox.attr_gene()
            
        # Validity Check: Only keep mutation if it produces a valid model
        if ess.is_valid_architecture(mutant):
            individual[:] = mutant # Apply change
            
        return individual, # DEAP requires a tuple return

    # ==========================================================================
    # PUBLIC INTERFACE (Use these in main.py)
    # ==========================================================================
    
    def initialize_population(self):
        """Creates Generation 0."""
        return self.toolbox.population(n=self.pop_size)

    def evolve_next_generation(self, population):
        """
        The Core Loop: Selection -> Crossover -> Mutation.
        """
        # 1. Select the next generation's parents
        offspring = self.toolbox.select(population, len(population))
        
        # 2. Clone them (Vital: DEAP modifies in-place)
        offspring = list(map(self.toolbox.clone, offspring))

        # 3. Apply Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cx_rate:
                self.toolbox.mate(child1, child2)
                # Invalidated fitness means they need to be re-evaluated
                del child1.fitness.values
                del child2.fitness.values

        # 4. Apply Mutation
        for mutant in offspring:
            if random.random() < self.mut_rate:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    # --- THE SAFETY NET (CHECKPOINTING) ---
    
    def save_checkpoint(self, population, generation, filepath):
        """
        Saves the entire state of the experiment to a file.
        If Colab crashes, we use this to resume.
        """
        data = {
            "population": population,
            "generation": generation,
            "rndstate": random.getstate()
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"Checkpoint saved successfully: {filepath}")

    def load_checkpoint(self, filepath):
        """
        Tries to load a previous state.
        Returns: (population, generation_number)
        """
        if not os.path.exists(filepath):
            print("No checkpoint found. Starting fresh.")
            return None, 0
            
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        random.setstate(data["rndstate"])
        print(f"Checkpoint loaded! Resuming from Generation {data['generation']}")
        return data["population"], data["generation"]