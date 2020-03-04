import copy
import util as ps

class PathCollection:
    def __init__(self, pd, noloops=True, directed=False):
        self._t = ps.cer if not directed else ps.identity

        self.pd = pd = copy.deepcopy(pd)

        self.endpoints = set()
        self.to_edge_ids = dict()
        self.from_edge_ids = dict()
        self.to_node_ids = dict()
        self.from_node_ids = dict()
        self.to_path_ids = dict()
        self.from_path_ids = dict()
        self.path_to_edge_ids = dict()
        self.path_to_node_ids = dict()

        for src in pd:
            for dst in pd[src]:
                if not noloops or src != dst:
                    path = pd[src][dst]
                    path_edges = ps.edgesin(path, directed=directed, trim=True)

                    node_ids = set()
                    for node in path:
                        node_id = self.to_node_ids.get(node, None)
                        if node_id is None:
                            node_id = len(self.from_node_ids)
                            self.to_node_ids[node] = node_id
                            self.from_node_ids[node_id] = node
                        node_ids.add(node_id)

                    edge_ids = set()
                    for edge in path_edges:
                        edge_id = self.to_edge_ids.get(edge, None)
                        if edge_id is None:
                            edge_id = len(self.from_edge_ids)
                            self.to_edge_ids[edge] = edge_id
                            self.from_edge_ids[edge_id] = edge
                        edge_ids.add(edge_id)

                    self.endpoints.add(self.to_node_ids[src])
                    self.endpoints.add(self.to_node_ids[dst])

                    path_id = len(self.from_path_ids)
                    self.to_path_ids[(src, dst)] = path_id
                    self.from_path_ids[path_id] = (src, dst)
                    self.path_to_edge_ids[path_id] = edge_ids
                    self.path_to_node_ids[path_id] = node_ids

    def __len__(self):
        return len(self.from_path_ids)

    def __iter__(self):
        return self.from_path_ids.__iter__()

    def get_edge_id_set(self):
        return set(self.from_edge_ids)
    
    def get_node_id_set(self):
        return set(self.from_node_ids)

    def get_path_id_set(self):
        return set(self.from_path_ids)

    def get_edge_ids_from_path(self, src, dst):
        return self.get_edge_ids_from_path_id(self.get_path_id(src, dst))

    def get_node_ids_from_path(self, src, dst):
        return self.get_node_ids_from_path_id(self.get_path_id(src, dst))

    def get_edge_id(self, edge):
        return self.to_edge_ids[self._t(edge)]
    
    def get_node_id(self, node):
        return self.to_node_ids[node]
    
    def get_endpoint_set(self):
        return set(self.endpoints)

    def get_path_id(self, src, dst):
        return self.to_path_ids[(src, dst)]

    def get_edge_ids_from_path_id(self, path_id):
        return self.path_to_edge_ids[path_id]

    def get_node_ids_from_path_id(self, path_id):
        return self.path_to_node_ids[path_id]
