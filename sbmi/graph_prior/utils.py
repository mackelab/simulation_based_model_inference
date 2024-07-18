import itertools
import torch
import numpy as np


def get_densly_connected_components(c_list, weight=None):
    """
    returns all "directed" pairs of c_list.
    c_list: list of node names
    weight: if not None: constant weight is appended to each tuple,
            resulting in (n1,n2,w) otherwise only (n1,n2) is returned
    """
    # initialize list of edges
    c1 = list(itertools.combinations(c_list, 2))
    # append reversed edges
    c2 = [c[::-1] for c in c1]
    [c1.append(c) for c in c2]
    if weight != None:
        c3 = [c + (weight,) for c in c1]
        return c3
    else:
        return c1


def calc_model_prior_sample(model_inds, ordered=False):
    """calculates the sample prior probabilities

    Args:
        model_inds (_type_): binary array of model indeces
        ordered (bool): return ordered tensors according to probs.

    Returns:
        _type_: probs and corresponding binary model vectors
    """
    n = model_inds.shape[0]
    n_components = model_inds.shape[1]
    lst = torch.tensor(
        [list(i) for i in itertools.product([0, 1], repeat=n_components)]
    )
    p = torch.zeros(len(lst))
    for i, item in enumerate(lst):
        p[i] = (model_inds == item).all(1).sum()
    p /= n

    lst = lst[p > 0]
    p = p[p > 0]

    if ordered:
        order = torch.argsort(p, descending=True)
    else:
        order = np.arange(len(p))

    return p[order], lst[order]
