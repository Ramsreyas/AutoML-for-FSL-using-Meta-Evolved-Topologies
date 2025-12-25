# main_base.py
import torch
import argparse
import sys
import os

# 1. Setup System Path so we can find 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# 2. Import your modules
from src.model_builder_base import build_baseline_cnn
from src.data_loader import create_task_generator
from src.fitness import evaluate_fitness

def run_experiment(experiment_name, ways, shots, device, data_root):
    print(f"\n{'='*60}")
    print(f"STARTING EXPERIMENT: {experiment_name}")
    print(f"Configuration: {ways}-Way {shots}-Shot")
    
    # Standard Omniglot Params
    QUERY_SAMPLES = 15      
    ADAPTATION_STEPS = 1    
    INNER_LR = 0.25         
    NUM_TASKS = 100  # Fast validation
    
    # LOAD DATA
    tasksets = create_task_generator(
        dataset_root=data_root, ways=ways, shots=shots, query_tasks=QUERY_SAMPLES
    )

    # BUILD MODEL (Baseline Conv-4)
    model = build_baseline_cnn(ways=ways, channels=1)
    model.to(device)

    # EVALUATE
    accuracy = evaluate_fitness(
        model=model, task_generator=tasksets, 
        ways=ways, shots=shots, query_samples=QUERY_SAMPLES,
        adaptation_steps=ADAPTATION_STEPS, inner_lr=INNER_LR, num_tasks_to_test=NUM_TASKS
    )

    print(f"RESULT >> {experiment_name}: {accuracy*100:.2f}% Accuracy")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/content/data', help='Path to dataset')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device}")

    # The 6-Point Stress Test
    experiments = [
        ("EXP_A_Standard_5W_1S", 5, 1),
        ("EXP_B_Standard_5W_5S", 5, 5),
        ("EXP_C_Intermed_10W_1S", 10, 1),
        ("EXP_D_Intermed_10W_5S", 10, 5),
        ("EXP_E_Extreme_20W_1S", 20, 1),
        ("EXP_F_Extreme_20W_5S", 20, 5),
    ]

    results = {}
    for name, ways, shots in experiments:
        acc = run_experiment(name, ways, shots, device, args.data_root)
        results[name] = acc

    print("\n" + "="*30 + "\nFINAL BENCHMARK REPORT\n" + "="*30)
    for name, acc in results.items():
        print(f"{name:<25} | {acc*100:.2f}%")