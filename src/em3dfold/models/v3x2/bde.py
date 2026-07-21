from collections import namedtuple

import torch
from torch import nn

from em3dfold.models.modules import SinusoidalPositionalEncoding
from em3dfold.models.knn_graph_torch import knn_graph
BackboneDistanceEmbeddingOutput = namedtuple(
    "BackboneDistanceEmbeddingOutput",
    [
        "pos3d_emb",
        "edge_index",
        "positions",
        "neighbour_positions",
        "full_edge_index",
    ],
)

class BackboneDistanceEmbedding(nn.Module):
    def __init__(
        self,
        num_neighbours: int = 32,
        position_encoding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.k = num_neighbours
        self.d_pe = position_encoding_dim
        self.distance_encoding = SinusoidalPositionalEncoding(self.d_pe)

    def forward(
        self,
        affines,
        edge_index = None,
        full_edge_index = None,
        batch = None,
        k = None,
    ):
        # Get positions from affines.
        positions = affines[..., :3, 3] # n 3

        if (edge_index is None) != (full_edge_index is None):
            raise ValueError(
                "edge_index and full_edge_index must either both be None or both be provided."
            )

        # Get edge index from knn graph if not provided.
        if edge_index is None:
            edge_index = knn_graph(
                positions,
                k,
                batch=batch,
                loop=False,
                flow="source_to_target",
            )
            full_edge_index = edge_index
            edge_index = edge_index[0].reshape(len(positions), k) # n k

        # Get position embeddings.
        position3d_embeddings = self.distance_encoding(positions).flatten(1) # n 3 * d_pe

        # Use global relative vectors so downstream cryo rectangle sampling
        # rotates in the same coordinate system as the map/grid coordinates.
        neighbour_positions = positions[edge_index] - positions[:, None]

        return BackboneDistanceEmbeddingOutput(
            pos3d_emb=position3d_embeddings,
            positions=positions,
            neighbour_positions=neighbour_positions,
            edge_index=edge_index,
            full_edge_index=full_edge_index,
        )

