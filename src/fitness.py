# src/fitness.py
# This file contains the logic for Module 4: The Fitness Evaluation Function.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import time

# Import our own helper function from another module
from .data_loader import split_task 

def evaluate_fitness(model, task_generator, ways, shots, query_samples, 
                     adaptation_steps, inner_lr, num_tasks_to_test):
    """
    Evaluates a model's fitness based on its meta-learning performance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    loss_function = nn.CrossEntropyLoss()
    total_accuracy = 0.0
    start_time = time.time()

    for task_idx in range(num_tasks_to_test):
        # 1. Clone model for this task
        temp_model = copy.deepcopy(model)
        temp_model.train()
        optimizer = optim.SGD(temp_model.parameters(), lr=inner_lr)

        # 2. Sample a new task
        task_batch = task_generator.train.sample()
        data, labels = task_batch
        data, labels = data.to(device), labels.to(device)

        # 3. Split into support and query sets (using our new helper)
        support_data, support_labels, query_data, query_labels = split_task(
            (data, labels), ways, shots, query_samples
        )

        # 4. Fast adaptation on support set
        for step in range(adaptation_steps):
            optimizer.zero_grad()
            outputs = temp_model(support_data)
            loss = loss_function(outputs, support_labels)
            loss.backward()
            optimizer.step()

        # 5. Evaluate on query set
        temp_model.eval()
        with torch.no_grad():
            query_outputs = temp_model(query_data)
            _, preds = torch.max(query_outputs, 1)
            correct = (preds == query_labels).sum().item()
            total_accuracy += correct / len(query_labels)

        # Progress log
        if (task_idx + 1) % 20 == 0:
            print(f"  ... evaluated task {task_idx + 1}/{num_tasks_to_test}")

    end_time = time.time()
    print(f"Evaluation of {num_tasks_to_test} tasks took {end_time - start_time:.2f} seconds.")

    # Average accuracy = fitness score
    final_fitness = total_accuracy / num_tasks_to_test
    return final_fitness