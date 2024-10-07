from deepgd.constants import DATASET_ROOT
from deepgd.data import BaseData, GraphDrawingData

import os
import re
import hashlib
from typing import Callable, Optional, TypeVar, Iterator

from tqdm.auto import tqdm
import numpy as np
import torch
import torch_geometric as pyg
import networkx as nx
import itertools

DATATYPE = TypeVar("DATATYPE", bound=BaseData)


class RomeDataset(pyg.data.InMemoryDataset):

    DEFAULT_NAME = "Rome"
    DEFAULT_URL = "https://www.graphdrawing.org/download/rome-graphml.tgz"
    GRAPH_NAME_REGEX = re.compile(r"grafo(\d+)\.(\d+)")

    def __init__(self, *,
                 url: str = DEFAULT_URL,
                 root: str = DATASET_ROOT,
                 name: str = DEFAULT_NAME,
                 index: Optional[list[str]] = None,
                 datatype: type[DATATYPE] = GraphDrawingData):
        print('inside init')
        self.url: str = url
        self.dataset_name: str = name
        self.index: Optional[list[str]] = index
        self.datatype: type[DATATYPE] = datatype
        super().__init__(
            root=os.path.join(root, name),
            transform=self.datatype.dynamic_transform,
            pre_transform=self.datatype.pre_transform,
            pre_filter=self.datatype.pre_filter
        )
        self.data, self.slices = torch.load(self.data_path)
        with open(self.index_path, "r") as index_file:
            self.index = index_file.read().strip().split("\n")
        data_list = map(datatype.static_transform, tqdm(self, desc=f"Transform graphs"))
        # total = 0
        # for _ in data_list:
        #     total += 1
        # print("datalist", total) #345
        data_dict = {data.G.graph["name"]: data for data in data_list}
        print('data dictionary size:', len(data_dict.keys()))
        #data_list = [data_dict[name] for name in self.index]
        # data_list = [data_dict[name] for name in ['path'+str(i) for i in range(5,10)]]
        node_sizes = [20, 30, 40, 50,80]
        probabilities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        data_list = []
        for node_size in node_sizes:
            for prob in probabilities:
                for i in range(10): 
                    graph_name = f'erdos_renyi_{node_size}_{prob}_graph_{i}'
                    if graph_name in data_dict: 
                        data_list.append(data_dict[graph_name])
                    else:
                        print(f"Graph {graph_name} not found")

        self.data, self.slices = self.collate(list(data_list))

    def _parse_metadata(self, logfile: str) -> Iterator[str]:
        print('inside _parse_metadata')
        with open(logfile) as fin:
            for line in fin.readlines():
                if match := self.GRAPH_NAME_REGEX.search(line):
                    yield match.group(0)

    @property
    def raw_file_names(self) -> list[str]:
        print('raw_file_names')
        metadata_file = "rome/Graph.log"
        if os.path.exists(metadata_path := os.path.join(self.raw_dir, metadata_file)):
            return list(map(lambda f: f"rome/{f}.graphml", self._parse_metadata(metadata_path)))
        return [metadata_file]

    @property
    def processed_file_names(self) -> list[str]:
        print('processed_file_names')
        return ["data.pt", "index.txt"]

    @property
    def data_path(self) -> str:
        return self.processed_paths[0]

    @property
    def index_path(self) -> str:
        return self.processed_paths[1]
    
    def generate(self) -> Iterator[nx.Graph]:
        print('inside generate')
        def key(path):
            match = self.GRAPH_NAME_REGEX.search(path)
            return int(match.group(1)), int(match.group(2))
        '''
        for file in tqdm(sorted(self.raw_paths, key=key), desc=f"Loading graphs"):
            G = nx.read_graphml(file)
            G.graph.update(dict(
                name=self.GRAPH_NAME_REGEX.search(file).group(0),
                dataset=self.dataset_name
            ))
            yield G
        '''
        #file  = self.raw_paths[0]
        #G = nx.read_graphml(file)
        #G.graph.update(dict(
        #    name=self.GRAPH_NAME_REGEX.search(file).group(0),
        #    dataset=self.dataset_name
        #))
        #yield G
        # for i in range(5, 10):
        #     G = nx.path_graph(i)
        #     G.graph.update(dict(
        #         name='path'+str(i),
        #         dataset='test'
        #     ))
        #     yield G
        node_sizes = [20, 30, 40, 50,80]
        probabilities = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        count = 0
        for node_size, prob in itertools.product(node_sizes, probabilities):
            for i in range(10):  # Generate 10 graphs per combination
                count += 1
                G = nx.erdos_renyi_graph(n=node_size, p=prob)
                G.graph.update(dict(
                    name=f'erdos_renyi_{node_size}_{prob}_graph_{i}',
                    dataset='test'
                ))
                if count>340:
                    print('count', count)
                yield G

                
    def download(self) -> None:
        print('inside download')
        pyg.data.download_url(self.url, self.raw_dir)
        pyg.data.extract_tar(f'{self.raw_dir}/rome-graphml.tgz', self.raw_dir)

    def process(self) -> None:
        print('inside process')
        def filter_and_save_index(data_list):
            name_list = []
            for data in data_list:
                if self.pre_filter(data):
                    name_list.append(data.G.graph["name"])
                    yield data
            if self.index is None:
                self.index = name_list
            else:
                self.index = [name for name in self.index if name in name_list]
            with open(self.index_path, "w") as index_file:
                index_file.write("\n".join(self.index))

        # print('before generate')
        data_list = map(self.datatype, self.generate())
        # print(" data_list = map(self.datatype, self.generate())" , len(list(map(self.datatype, self.generate()))))
        data_list = filter_and_save_index(data_list)
        # print("data_list = filter_and_save_index(data_list)",len(list(filter_and_save_index(data_list))))
        data_list = map(self.pre_transform, data_list)
        # print("data_list = map(self.pre_transform, data_list)",len(list(map(self.pre_transform, data_list))))
        data, slices = self.collate(list(data_list))
        
        torch.save((data, slices), self.processed_paths[0])
