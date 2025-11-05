# src/data_loader.py
# This file contains the logic for Module 1: The Meta-Learning Environment.

import learn2learn as l2l
import numpy as np

def create_task_generator(dataset_root, ways, shots, query_tasks):
    """
    Creates a learn2learn task generator for the Omniglot dataset.

    Args:
        dataset_root (str): The directory to store the dataset.
        ways (int): The number of classes per task (N-way).
        shots (int): The number of training examples per class (K-shot).
        query_tasks (int): The number of test examples per class in the query set.

    Returns:
        learn2learn.data.TaskDataset: The task generator object.
    """
    print(f"Loading Omniglot dataset from: {dataset_root}")
    tasksets = l2l.vision.benchmarks.get_tasksets(
        'omniglot',
        train_ways=ways,
        train_samples=shots + query_tasks, # Total samples per class
        test_ways=ways,
        test_samples=shots + query_tasks,
        root=dataset_root,
    )
    print("Dataset loaded successfully.")
    return tasksets

def split_task(task_batch, ways, shots, query_samples):
    """
    Splits a raw data batch from the task generator into support and query sets.
    
    Args:
        task_batch (tuple): A (data, labels) tuple from task_generator.sample().
        ways (int): N_WAY
        shots (int): K_SHOT
        query_tasks (int): Number of query samples per class.

    Returns:
        (tuple): (support_data, support_labels, query_data, query_labels)
    """
    data, labels = task_batch
    total_samples_per_class = shots + query_samples
    
    support_indices = []
    query_indices = []

    for i in range(ways):
        start = i * total_samples_per_class
        end = start + total_samples_per_class
        
        # First K_SHOT go to Support
        support_indices.extend(range(start, start + shots))
        
        # Remaining QUERY_SAMPLES go to Query
        query_indices.extend(range(start + shots, end))

    # We use integer array indexing, which is faster than boolean masking.
    support_data = data[support_indices]
    support_labels = labels[support_indices]
    query_data = data[query_indices]
    query_labels = labels[query_indices]
    
    return support_data, support_labels, query_data, query_labels