import numpy as np


def get_batch_size_overground_sample_percentage(
    world_size: int, my_rank: int, global_batch_size: int, oversample_foreground_percent
):

    batch_sizes = []
    oversample_percents = []

    batch_size_per_GPU = np.ceil(global_batch_size / world_size).astype(int)

    for rank in range(world_size):
        if (rank + 1) * batch_size_per_GPU > global_batch_size:
            batch_size = batch_size_per_GPU - (
                (rank + 1) * batch_size_per_GPU - global_batch_size
            )
        else:
            batch_size = batch_size_per_GPU

        batch_sizes.append(batch_size)

        sample_id_low = 0 if len(batch_sizes) == 0 else np.sum(batch_sizes[:-1])
        sample_id_high = np.sum(batch_sizes)

        if sample_id_high / global_batch_size < (1 - oversample_foreground_percent):
            oversample_percents.append(0.0)
        elif sample_id_low / global_batch_size > (1 - oversample_foreground_percent):
            oversample_percents.append(1.0)
        else:
            percent_covered_by_this_rank = (
                sample_id_high / global_batch_size - sample_id_low / global_batch_size
            )
            oversample_percent_here = 1 - (
                (
                    (1 - oversample_foreground_percent)
                    - sample_id_low / global_batch_size
                )
                / percent_covered_by_this_rank
            )
            oversample_percents.append(oversample_percent_here)

    print("worker", my_rank, "oversample", oversample_percents[my_rank])
    print("worker", my_rank, "batch_size", batch_sizes[my_rank])

    return int(batch_sizes[my_rank]), oversample_percents[my_rank]
