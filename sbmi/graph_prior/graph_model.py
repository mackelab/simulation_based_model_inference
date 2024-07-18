import copy
import networkx as nx
import numpy as np


class ModelGraph:
    """
    class for the model graph
    0 is starting node.
    -1 is end node.
    """

    def __init__(
        self,
        model_dict,
        unique_components,
        edges,
        node_priors,
        pos_coupling=None,
        neg_coupling=None,
        exclusive_nodes=None,
        pos_coupling_factor=2.0,
    ):
        """_summary_

        Args:
            model_dict (dict):
            unique_components (dict): each entry a boolean indicating if the key node is unique
            edges (list): list defining the (initial) edges of the graph prior. format: (node_1,node_2,weight_12)
            node_priors (dict): dictionary of dictionaries. for each node specifying the priors for all variables c_x.
            pos_coupling (dict, optional): each entry a list of nodes for which the key has positive coupling.
                Defaults to None.
            neg_coupling (dict, optional): each entry a list of nodes for which the key has negative coupling.
                Defaults to None.
            exclusive_nodes (dict, optional): defining mutually exclusive_nodes. Defaults to None.
            pos_coupling_factor (float): upgrading factor for positive coupling,
                negative_coupling_factor is then defined by 1/pos_coupling_factor
        """
        self.model_dict = model_dict
        self.unique_components = unique_components
        self.node_priors = node_priors

        # dynamic properties
        self.pos_coupling = pos_coupling
        self.neg_coupling = neg_coupling
        self.exclusive_nodes = exclusive_nodes
        self.pos_coupling_factor = pos_coupling_factor
        self.neg_coupling_factor = 1 / self.pos_coupling_factor

        # construct graph
        self.G = nx.DiGraph()
        # self.edges = self.G.edges
        # add start and end nodes
        self.G.add_node(0)
        self.G.add_node(-1)
        self.G.nodes[-1]["unique"] = True
        # add looping end node
        self.add_edge(-1, -1, 1)

        # add nodes
        for key in self.model_dict.keys():
            self.G.add_node(key)

        # add edges
        for edge in edges:
            self.add_edge(*edge)

        # add node priors and dynamic properties
        for key in self.node_priors.keys():
            self.G.nodes[key]["prior"] = self.node_priors[key]
            self.G.nodes[key]["unique"] = self.unique_components[key]
            if self.pos_coupling != None:
                self.G.nodes[key]["pos_coupling"] = self.pos_coupling[key]
            if self.neg_coupling != None:
                self.G.nodes[key]["neg_coupling"] = self.neg_coupling[key]
            if self.exclusive_nodes != None:
                self.G.nodes[key]["exclusive_nodes"] = self.exclusive_nodes[key]

    @property
    def edges(self):
        return [
            (e + (self.G.get_edge_data(list(e)[0], list(e)[1])["weight"],))
            for e in self.G.edges
        ]

    def add_edge(self, n1, n2, weight=1):
        self.G.add_edge(n1, n2, weight=weight)

    def remove_edge(self, n1, n2):
        self.G.remove_edge(n1, n2)

    def get_random_walk(self, max_walk_length=1000, starting_node=0, return_p=False):
        """
        starting node is still an int !!! not node name.
        ---
        max_walk_length (int): either sampling rw of this length or reaching an end node (-1)
        """

        # make a deep copy as we will potentially change the graph during the random walk
        G1 = copy.deepcopy(self.G)

        # get adjecency matrix
        # A = nx.adjacency_matrix(self.G).todense()
        # A = np.array(A, dtype = np.float64)
        A = nx.to_numpy_array(G1, dtype=np.float64)

        # normalize it
        division_const = np.sum(A, 1)[:, None]
        T = A / division_const

        # initialize starting node
        p = np.zeros(len(T[0]))
        p[starting_node] = 1

        """
        if max_walk_length == None:
            visited = []
            p_visited = []
            current_visit = starting_node
            visited.append(starting_node)
            current_visit_name = np.array(self.G)[current_visit]
            while current_visit_name != -1:
                p = np.dot(T.T, p)
                current_visit = int(np.random.choice(np.arange(len(p)), p=p))
                visited.append(current_visit)
                p_visited.append(p[current_visit])
                p = np.zeros(len(T[0]))
                p[current_visit] = 1
                current_visit_name = np.array(self.G)[current_visit]

                # update T for unique nodes
                if self.G.nodes[current_visit_name]["unique"]:
                    T[:, current_visit] = 0
                    # normalize it
                    division_const = np.sum(T, 1)[:, None]
                    division_const[division_const == 0] = 1
                    T = T / division_const
        else:
        """

        visited = np.zeros(max_walk_length + 1, dtype=int)
        visited[0] = starting_node
        p_visited = np.zeros(max_walk_length)
        for k in range(max_walk_length):
            p = np.dot(T.T, p)
            visited[k + 1] = int(np.random.choice(np.arange(len(p)), p=p))
            p_visited[k] = p[visited[k + 1]]
            p = np.zeros(len(T[0]))
            p[visited[k + 1]] = 1
            current_visit_name = np.array(G1)[visited[k + 1]]
            if current_visit_name == -1:
                visited = visited[: k + 2]
                p_visited = p_visited[: k + 2]
                break

            """
            # update T for unique nodes
            if self.G.nodes[current_visit_name]["unique"]:
                T[:, visited[k + 1]] = 0
                # normalize it
                division_const = np.sum(T, 1)[:, None]
                division_const[division_const == 0] = 1
                T = T / division_const
            """
            G1, T = self.update_dynamic_properties(G1, current_visit_name, visited, k)

        # re-index to original labels
        visited = np.array(self.G)[visited]

        if return_p:
            return visited, p_visited
        else:
            return visited

    def update_dynamic_properties(self, G1, current_visit_name, visited, k):
        # update T for unique nodes
        if G1.nodes[current_visit_name]["unique"]:
            # T[:, visited[k + 1]] = 0
            # normalize it
            # division_const = np.sum(T, 1)[:, None]
            # division_const[division_const == 0] = 1
            # T = T / division_const

            incoming_nodes = list(G1.predecessors(current_visit_name))
            for node_in in incoming_nodes:
                G1.remove_edge(node_in, current_visit_name)

        # update other dynamic properties
        if self.pos_coupling != None:
            for node in G1.nodes[current_visit_name]["pos_coupling"]:
                incoming_nodes = list(G1.predecessors(node))
                for node_in in incoming_nodes:
                    weight_old = G1.get_edge_data(node_in, node)["weight"]
                    G1.add_edge(
                        node_in, node, weight=weight_old * self.pos_coupling_factor
                    )

        if self.neg_coupling != None:
            for node in G1.nodes[current_visit_name]["neg_coupling"]:
                incoming_nodes = list(G1.predecessors(node))
                for node_in in incoming_nodes:
                    weight_old = G1.get_edge_data(node_in, node)["weight"]
                    G1.add_edge(
                        node_in, node, weight=weight_old * self.neg_coupling_factor
                    )

        if self.exclusive_nodes != None:
            for node in G1.nodes[current_visit_name]["exclusive_nodes"]:
                incoming_nodes = list(G1.predecessors(node))
                for node_in in incoming_nodes:
                    G1.remove_edge(node_in, node)

        T = nx.to_numpy_array(G1, dtype=np.float64)
        # normalize it
        division_const = np.sum(T, 1)[:, None]
        division_const[division_const == 0] = 1
        T = T / division_const

        return G1, T

    def get_node_samples(self, random_walk):
        """
        sample from priors for each node of the random walk.
        assuming that the rw starts with a dummy node 0.
        """
        samples = [{}]
        for i, node in enumerate(random_walk):
            if node != 0 and node != -1:
                node_samples = dict.fromkeys(self.G.nodes[node]["prior"])
                for j, item in enumerate(self.G.nodes[node]["prior"]):
                    node_samples[item] = self.G.nodes[node]["prior"][item]()
                samples.append(node_samples)

        if len(samples) < len(random_walk):
            samples.append({})

        # check if we have samples for each node
        assert len(samples) == len(random_walk)

        return samples

    def get_constants(self, const_name="c_", n_digits=3):
        """
        extracts constants from formula (as string) model components
        constants should start with const_name and contain exactly n_digits digits
        """
        constants = dict.fromkeys(self.model_dict.keys())

        for key, item in self.model_dict.items():
            constants[key] = []
            i = 0
            position = 0
            while i >= 0:
                i = item[position:].find(const_name)
                if i >= 0:
                    constants[key].append(item[i + position : i + position + n_digits])
                position = position + i + 1
        return constants


"""
sample complete model
"""


class Model:
    """
    base class for different models.
    -----
    assuming that node_prior_bounds are used in the node priors as a UNIFORM distribution.
    """

    def __init__(
        self,
        model_dict,
        unique_components,
        node_priors,
        edges,
        variables,
        CompileClass,
        node_prior_bounds,
        theta_partition,
        pos_coupling=None,
        neg_coupling=None,
        exclusive_nodes=None,
        pos_coupling_factor=2.0,
    ):
        """_summary_

        Args:
            model_dict (_type_): _description_
            unique_components (_type_): _description_
            node_priors (_type_): _description_
            edges (_type_): _description_
            variables (_type_): _description_
            CompileClass (_type_): _description_
            node_prior_bounds (Tensor): upper and lower limits for node_priors (uniform distributionl)
        """
        self.model_dict = model_dict
        self.unique_components = unique_components
        self.node_priors = node_priors
        self.variables = variables
        self.model_graph = ModelGraph(
            self.model_dict,
            self.unique_components,
            edges,
            self.node_priors,
            pos_coupling=pos_coupling,
            neg_coupling=neg_coupling,
            exclusive_nodes=exclusive_nodes,
            pos_coupling_factor=pos_coupling_factor,
        )
        self.CompileClass = CompileClass
        self.node_prior_bounds = node_prior_bounds
        self.theta_partition = theta_partition
        self.total_params = node_prior_bounds.shape[1]
        self.n_components = len(self.theta_partition)

        assert theta_partition.sum() == self.total_params

        assert len(self.model_dict) == len(self.node_priors)

    @property
    def edges(self):
        return self.model_graph.edges

    def sample_model_skeleton(
        self, max_walk_length=None, starting_node=0, return_p=False
    ):
        return self.model_graph.get_random_walk(
            max_walk_length, starting_node, return_p
        )

    def sample_constants(self, model_ind):
        return self.model_graph.get_node_samples(model_ind)

    def compile_model(self, model_ind, constants):
        raise Exception("Not Implemented")

    def run_model(self, model):
        raise Exception("Not Implemented")

    def sample_model(self, max_len_token=1_000):
        model_ind = self.sample_model_skeleton(max_walk_length=max_len_token)
        constants = self.sample_constants(model_ind)
        model = self.compile_model(model_ind, constants)

        return model
