from deepgd.data import BaseData, GraphDrawingData
import torch
import networkx as nx
from torch_geometric.data import InMemoryDataset, Data
from tqdm.auto import tqdm
import numpy as np
from typing import Optional, TypeVar, List, Iterator
import os

DATATYPE = TypeVar("DATATYPE", bound=BaseData)


class ErdosRenyiDataset(InMemoryDataset):
    DEFAULT_NAME = "ErdosRenyi"

    def __init__(self, *,
                 root: str,
                 name: str = DEFAULT_NAME,
                 node_sizes: List[int] = [20, 40, 80],
                 probabilities: List[float] = [0.2, 0.4, 0.6, 0.8],
                 num_graphs_per_combination: int = 10,
                 index: Optional[list[str]] = None,
                 datatype: type[DATATYPE] = GraphDrawingData):
        self.dataset_name: str = name
        self.node_sizes = node_sizes
        self.probabilities = probabilities
        self.num_graphs_per_combination = num_graphs_per_combination
        self.index: Optional[list[str]] = index
        self.datatype: type[DATATYPE] = datatype

        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter
        )

        # Generate graphs and prepare data
        data_list = map(datatype.static_transform, tqdm(self, desc=f"Transform graphs"))
        data_list = list(data_list)
        self.data, self.slices = self.collate(data_list)

    def generate(self) -> Iterator[nx.Graph]:
        """
        Generate Erdős-Rényi graphs for all combinations of node sizes and probabilities.
        """
        graph_counter = 0
        for num_nodes in self.node_sizes:
            for p in self.probabilities:
                for i in range(self.num_graphs_per_combination):
                    G = nx.erdos_renyi_graph(num_nodes, p)
                    G.graph.update(dict(
                        name=f"erdos_renyi_{num_nodes}_{p}_{i}",
                        dataset=self.dataset_name
                    ))
                    graph_counter += 1
                    yield G

    @property
    def raw_file_names(self) -> list[str]:
        """
        Dummy property to maintain compatibility with InMemoryDataset structure.
        """
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]

    def process(self) -> None:
        """
        Processes the generated Erdős-Rényi graphs and stores the data.
        """
        def filter_and_save_index(data_list):
            name_list = []
            for data in data_list:
                if self.pre_filter(data):
                    name_list.append(data.G.graph["name"])
                    yield data
            self.index = name_list
            with open(self.index_path, "w") as index_file:
                index_file.write("\n".join(self.index))

        data_list = map(self.datatype, self.generate())
        data_list = filter_and_save_index(data_list)
        data_list = map(self.pre_transform, data_list)
        data, slices = self.collate(list(data_list))
        torch.save((data, slices), self.processed_paths[0])


'''
# Step 1: Initialize the dataset
dataset = ErdosRenyiDataset(
    root="path/to/dataset",
    node_sizes=[20, 40, 80],  # Node sizes as specified
    probabilities=[0.2, 0.4, 0.6, 0.8],  # Probabilities as specified
    num_graphs_per_combination=10,  # 10 graphs per combination
    datatype=GraphDrawingData  # Same as before
)
'''



