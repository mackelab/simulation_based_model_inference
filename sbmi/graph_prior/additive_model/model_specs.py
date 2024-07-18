import numpy as np
import torch

from sbmi.graph_prior.additive_model.additive_model_compiler import (
    AdditiveModelGraph,
    CompiledAdditiveModel,
)
from sbmi.graph_prior.utils import get_densly_connected_components

"""
define model components
"""

model_dict = {
    1: "c_1*x_1",
    2: "c_1*x_1",
    3: "c_1*x_1*x_1",
    4: "c_1*sin(c_2*x_1)",
    100: "observer_noise",
    # 102: "correlated_noise",
    # 101 "initial_noise",
    103: "increasing_noise",
}


additive_model_ids = [1, 2, 3, 4]

unique_components = {
    1: True,
    2: True,
    3: True,
    4: True,
    100: True,
    103: True,
}

pos_coupling_factor = 2

pos_coupling = {
    1: [100, 103],
    2: [100, 103],
    3: [100, 103],
    4: [100, 103],
    100: [],
    103: [],
}

neg_coupling = {
    1: [2],  # neg. coupling for linear terms
    2: [1],
    3: [],
    4: [],
    100: [],
    103: [],
}

exclusive_nodes = {
    1: [],
    2: [],
    3: [],
    4: [],
    100: [],
    103: [],
}

theta_partition = torch.tensor([1, 1, 1, 2, 1, 1])

node_prior_bounds = torch.tensor(
    [[-2, -2, -0.5, 0, 0.5, 0.1, 0.5], [2, 2, 0.5, 5, 5, 2, 2]]
)

# Uniform priors
node_priors = {
    1: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 0], node_prior_bounds[1, 0]
        ),
    },
    2: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 1], node_prior_bounds[1, 1]
        ),
    },
    3: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 2], node_prior_bounds[1, 2]
        ),
    },
    4: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 3], node_prior_bounds[1, 3]
        ),
        "c_2": lambda: np.random.uniform(
            node_prior_bounds[0, 4], node_prior_bounds[1, 4]
        ),
    },
    100: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 5], node_prior_bounds[1, 5]
        )
    },
    103: {
        "c_1": lambda: np.random.uniform(
            node_prior_bounds[0, 6], node_prior_bounds[1, 6]
        )
    },
}


# structure: node_1, node_2, weight

# get all edges for additive nodes
edges = get_densly_connected_components([1, 2, 3, 4], weight=1)

# and add other edges:
# starting edges,
# noise edges
# leaves

edges2 = [
    (0, 1, 2),
    (0, 2, 1),  # less weight for l2
    (0, 3, 2),
    (0, 4, 2),
    (1, 100, 1),
    (2, 100, 1),
    (3, 100, 1),
    (4, 100, 1),
    (1, 103, 1),
    (2, 103, 1),
    (3, 103, 1),
    (4, 103, 1),
    (100, -1, 1),
    (103, -1, 1),
]

[edges.append(e) for e in edges2]

variables = ["x_1"]

"""
define resolution for equidistant time points
"""
dt = 0.02


additivemodel = AdditiveModelGraph(
    model_dict,
    unique_components,
    node_priors,
    edges,
    variables,
    additive_model_ids,
    CompileClass=CompiledAdditiveModel,
    node_prior_bounds=node_prior_bounds,
    theta_partition=theta_partition,
    pos_coupling=pos_coupling,
    neg_coupling=neg_coupling,
    exclusive_nodes=exclusive_nodes,
    pos_coupling_factor=pos_coupling_factor,
    dt=dt,
)
