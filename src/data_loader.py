# src/data_loader.py
# This file contains the logic for Module 1: The Meta-Learning Environment.
import learn2learn as l2l

# This function is the core of our M1 module. It's clean, reusable, and documented.
def create_task_generator(dataset_root, ways, shots, query_tasks):
    """
    Creates a learn2learn task generator for the Omniglot dataset.

    This function handles the download and setup of the Omniglot benchmark,
    wrapping it in a TaskDataset for easy sampling of few-shot learning tasks.

    Args:
        dataset_root (str): The directory to store the dataset.
        ways (int): The number of classes per task (N-way).
        shots (int): The number of training examples per class (K-shot).
        query_tasks (int): The number of test examples per class in the query set.

    Returns:
        learn2learn.data.TaskDataset: The task generator object.
    """
    print("Loading Omniglot dataset (will download if not present)...")
    # This is the key learn2learn function that creates the benchmark tasksets.
    tasksets = l2l.vision.benchmarks.get_tasksets(
        'omniglot',
        train_ways=ways,
        train_samples=shots + query_tasks, # Total samples needed per class for one task
        test_ways=ways,
        test_samples=shots + query_tasks,
        root=dataset_root,
    )
    print("Dataset loaded successfully.")
    return tasksets