"""
Shared model utilities.
"""
import torch


def get_center_labels(batch) -> torch.Tensor:
    """
    Extract center node labels from a batched graph.

    For ego-graphs, center node is always at the start of each graph
    (index 0 within each sub-graph, by construction in graph_builder.py).

    Args:
        batch: PyG Batch object

    Returns:
        Tensor of center node labels [batch_size]
    """
    if hasattr(batch, 'ptr'):
        center_indices = batch.ptr[:-1]
    else:
        # Fallback for older PyG versions
        batch_size = batch.batch.max().item() + 1
        center_indices = []
        for i in range(batch_size):
            mask = (batch.batch == i)
            first_idx = mask.nonzero(as_tuple=True)[0][0]
            center_indices.append(first_idx)
        center_indices = torch.tensor(center_indices, device=batch.y.device)

    return batch.y[center_indices]
