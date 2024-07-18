import copy
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from sbmi.graph_prior.additive_model.utils import round_expr


def get_dataset(
    model_class,
    n=10,
    valid_inds=(0, None),
    sort=False,
    max_len_token=None,
    model_d="additive",
):
    """
    samples a dataset from model_class with n samples

    ---
    n: batchsize
    valid_inds: tuple, start and end index of samples.
    sort: if True sort model indeces for each model in increasing order,
            assuming model inds are unstructured
    model_d (str): used model, description in {'additive','ddm', 'HH'}
    returns: model_inds, constants, x, y
    """

    model = model_class.sample_model(max_len_token=max_len_token)

    model_inds = [[] for _ in range(n)]
    constants = [[] for _ in range(n)]

    if model_d == "additive":
        x0, y0 = model.run_model()
        y = np.zeros((n,) + y0.shape)

    elif model_d == "ddm":
        y0 = model.run_model()
        x0 = np.nan
        y = np.zeros((n,) + y0.shape)

    elif model_d == "HH":
        y0, x0 = model.run_model(return_stim=True)
        y = np.zeros((n,) + y0.shape)

    else:
        raise NotImplementedError("not implemented model class. ")

    del model
    for i in range(n):
        model = model_class.sample_model(max_len_token=max_len_token)
        model_inds[i] = model.model_ind  # [valid_inds[0]:valid_inds[1]]
        constants[i] = model.constants  # [valid_inds[0]:valid_inds[1]]
        y[i] = model.run_model(return_t=False)

        if 0 in model_inds[i]:
            model_inds[i] = model_inds[i][valid_inds[0] :]
            constants[i] = constants[i][valid_inds[0] :]

        if -1 in model_inds[i]:
            model_inds[i] = model_inds[i][: valid_inds[1]]
            constants[i] = constants[i][: valid_inds[1]]

        if sort:
            raise (Warning)("Not implemented. also constants need to get sorted.")
            model_inds[i] = np.sort(model_inds[i])

        # remove indeces
        # for ind in delete_indeces:
        #    if ind in model_inds[i]:
        #        item = list(model_inds[i])
        #        item.remove(ind)
        #        model_inds[i] = np.array(item)
        del model

    return model_inds, constants, x0, y


def post_process_samples(
    model_inds,
    constants,
    y,
    n_components,
    partition,
    unique_tokens,
    max_params=2,
    constant_names=["c_1", "c_2"],
    model_d=None,
):
    """transforms samples from get_dataset() to:
        binary model indeces
        flattened tensor of constants (batch,n_params_max)
        torch tensor

    Args:
        model_inds (_type_): model inds as list of int numbers
        constants (list): model constants as list of array of dict
        y (np.array): output of model
        n_components (int): max number of model components
        partition:  partition of the model
        unique_tokens ([str] or array of tokens (integers) ): if ['infere'] unique tokens get inferred from model_inds
            otherwise the list of unique tokens will be kept and just sorted.
        model_d: None or 'HH'. model description

    Returns:
        torch tensors: theta, binary_model, y

    """

    model_inds, constants = sort_model_components_constants(model_inds, constants)
    # y = y.clone().detach().double().squeeze()
    if model_d == "HH":
        y = torch.from_numpy(y).squeeze()
    else:
        y = torch.tensor(y, dtype=torch.double).squeeze()

    transformed_model_inds = transform_model_inds(
        model_inds, unique_tokens=unique_tokens
    )

    model_data = prepare_constants(
        transformed_model_inds,
        constants,
        n_components=n_components,
        max_params=max_params,
        constant_names=constant_names,
    )
    binary_model = torch.tensor(model_data[:, :, 0])

    theta = torch.tensor(model_data[:, :, 1:])

    theta = flatten_theta(theta, partition)

    return theta, binary_model, y


class BasicDataset(Dataset):
    """dataset class for x,y data. where x,y are arrays"""

    def __init__(self, x, y, include_ids=False, transform=None):
        """
        Args:
            x: tensor (n_samps, sample_shape )
            y: labels as tensor
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.x = x
        self.y = y
        self.transform = transform
        self.include_ids = include_ids
        if self.include_ids:
            self.ids = np.arange(len(self.x))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data = self.x[idx]
        label = self.y[idx]
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {"x": data, "y": label}

        if self.include_ids:
            sample["ids"] = self.ids[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_unique_tokens(model_inds):
    """
    returns sorted array of unique tokens from the input list of tokens
    """
    tokens_u = model_inds[0]
    for token in model_inds:
        tokens_u = np.concatenate((tokens_u, token))
        tokens_u = np.unique(tokens_u)

    return np.sort(tokens_u)


def transform_model_inds(model_inds, unique_tokens=["infere"]):
    """
    transforms model indeces to a binary vector,
    ith entry coding for presence/absence of ith model component
    model_inds: list of lists of model indices
    unique_tokens ([str] or array of tokens (integers) ): if ['infere'] unique tokens get inferred from model_inds
       otherwise the list of unique tokens will be kept and just sorted.
    returns: array with dim: (n_models, n_components)
    """
    if unique_tokens[0] == "infere":
        unique_tokens = get_unique_tokens(model_inds)
    else:
        unique_tokens = np.sort(unique_tokens)

    n = len(unique_tokens)
    y = np.zeros((len(model_inds), n))  # dim(n_models, n_components)
    for i, item in enumerate(unique_tokens):
        for n, token in enumerate(model_inds):
            if item in token:
                y[n, i] = 1
    return y


def invert_transform_model_inds(
    binary_model_ind, model_inds=[1, 2, 3, 4, 100, 101], threshold=0.5
):
    """
    transforms binary (or more general float numbers) model indeces to a original model_inds
    binary_model_ind: 1d array
    model_inds: original model indices from model graph

    """
    m = []
    for i, item in enumerate(model_inds):
        if binary_model_ind[i] > threshold:
            m.append(item)
    return np.array(m)


def sort_model_components_constants(model_inds, constants):
    """
    sorts for each complete model the model indices and the constants
    WARNING: this is an inplace operation
    """
    model_inds = copy.deepcopy(model_inds)

    for i, model in enumerate(model_inds):
        sorting = np.argsort(model)
        model.sort()
        constants[i] = np.array(constants[i], dtype=object)[sorting]

    return model_inds, constants


def prepare_constants(
    transformed_model_inds,
    constants,
    n_components,
    max_params=2,
    constant_names=["c_1", "c_2"],
):
    """
    puts the constants into an array, together with the binary model indices.
    impute nans to non-present model components
    max_params: maximal number of model constants
    constant_names: same constant names used for all model components
    """
    # check if all params are included with names
    assert len(constant_names) == max_params

    n = len(constants)
    c_array = np.zeros((n, n_components, max_params + 1)) * np.nan
    c_array[:, :, 0] = transformed_model_inds
    model_c_nr = 0
    for i, c in enumerate(constants):
        model_c_nr = 0
        for n_component in range(n_components):
            if transformed_model_inds[i, n_component] == 1:
                for j, c_name in enumerate(constant_names):
                    if c_name in c[model_c_nr].keys():
                        c_array[i, n_component, j + 1] = c[model_c_nr][c_name]

                model_c_nr += 1

    return c_array


def invert_prepare_constants(
    model_array, model_mask, max_params=2, constant_names=["c_1", "c_2"], threshold=0.5
):
    """
    puts the constants from an array, into a dict with the names defined in constant_names
    model_array: (n_components,max_params+1) with binary model in first dim
    model_mask: (n_components,max_params) binary mask indicating which theta is used in which model component
    max_params: maximal number of model constants
    constant_names: same constant names used for all model components
    returns: list of dict
    """
    # check if all params are included with names
    assert len(constant_names) == max_params
    assert model_array.shape[1] == max_params + 1

    model_array = np.array(model_array)

    m = []
    j = -1
    for i in range(model_array.shape[0]):  # iterate over model components
        if model_array[i, 0] > threshold:
            m.append({})
            j += 1
            for c in range(max_params):
                if model_mask[i, c] == 1:
                    m[j][constant_names[c]] = model_array[i, c + 1]
    return m


def flatten_theta(theta, partition):
    """flattens theta following the partition

    Args:
        theta (_type_): (batch, n_model_components, max_num_params)
        partition (_type_): _description_
    returns: theta (batch, total_params)
    """
    total_params = partition.sum()
    batch, n_model_components, _ = theta.shape
    max_params = partition.max()

    theta = theta.view(batch, n_model_components * max_params)

    # creat mask
    mask = torch.zeros(n_model_components, max_params, dtype=bool)
    for i in range(n_model_components):
        mask[i, : partition[i]] = 1

    mask = mask.flatten()

    return theta[:, mask]


"""
to reconstruct model from dataset
"""


def compile_model(
    model,
    model_pred,
    theta_pred,
    model_inds,
    partition,
    max_params=2,
    constant_names=["c_1", "c_2"],
    verbose=False,
    **kwargs,
):
    """_summary_

    Args:
        model (_type_): model class
        model_pred (_type_): model predictions binary
        theta_pred (_type_): theta predictions
        model_inds (_type_): model indices internally used
        partition (_type_):  partition of theta
        verbose (bool, optional): . Defaults to False.

    Returns:
        _type_: model_rebuild, model_ind_2
    """

    assert len(model_pred) == len(model_inds)

    # create model mask
    model_mask = get_model_mask(partition)

    # put thetas in array form
    theta_pred = unflatten_theta(theta_pred, partition)

    # for prediction
    model_ind_2 = invert_transform_model_inds(
        model_pred.squeeze(), model_inds=model_inds
    )
    constants_2 = invert_prepare_constants(
        torch.cat((model_pred.unsqueeze(1), theta_pred), 1).detach(),
        model_mask,
        constant_names=constant_names,
        max_params=max_params,
    )

    if verbose:
        print("given predictions:")
        print("model inds :", model_ind_2)
        print("model constants:", constants_2)

    model_rebuild = model.compile_model(model_ind_2, constants_2, **kwargs)

    return model_rebuild, model_ind_2


def compile_and_run_model(
    model,
    model_pred,
    theta_pred,
    model_inds,
    partition,
    n=1,
    len_t=1000,
    verbose=False,
    return_model=False,
    model_type="additive",
    n_trial=100,
    max_params=2,
    constant_names=["c_1", "c_2"],
    **kwargs,
):
    """compiles and runs the model based on the prediction for the model components and parameters theta
    Args:
        model (model_class): class of the model (for ex. additive_model)
        model_pred (tensor): binary model tensor (n_model_components)
        theta_pred (tensor): predicted parameters theta, (total_params)
                             with nans inserted for inactive components
        model_inds (list): list of internally used used model indices e.g. [1,2,100,101]
        partition (array):
        n (int): number of model evaluations
        len_t (int): len of one model eval for additive model.
        model_type (str): in ['additive', 'ddm'] type of the model, Default: "additive"
        n_trial(int): default=100. number of trials for ddm.
    """

    if model_type == "additive":
        try:
            # initialize with used model components
            model_rebuild, model_ind_2 = compile_model(
                model, model_pred, theta_pred, model_inds, partition, verbose=verbose
            )
            t, y = model_rebuild.run_model(n=n)
            if verbose:
                print("predicted formula: ", round_expr(model_rebuild.formula, 3))

        except:
            y = np.zeros((n, len_t)) * np.nan
            t = np.zeros(len_t) * np.nan
            model_rebuild = None
            # if verbose:
            warnings.warn("No valid model predicted!")
            if verbose:
                print("no valid model. ")
        if len(y.shape) != 2:
            y = np.zeros((n, len_t)) * np.nan

        if return_model:
            return t, y.squeeze(), model_rebuild
        else:
            return t, y.squeeze()

    elif model_type == "ddm":
        raise NotImplementedError("not implemented yet")
        # # try:
        # model_rebuild, model_ind_2 = compile_model(
        #     model,
        #     model_pred,
        #     theta_pred,
        #     model_inds,
        #     partition,
        #     verbose=verbose,
        #     **kwargs,
        # )
        # x = model_rebuild.run_model(n=n, n_trial=n_trial)
        # # except:
        # #    x = torch.zeros((n, n_trial, 2))
        # #    # if verbose:
        # #    warnings.warn("No valid model predicted!")
        # #    print("no valid model predicted.")
        # #    print("predicted inds:", model_ind_2)
        # if return_model:
        #     return x, model_rebuild
        # else:
        #     return x

    elif model_type == "hh":
        raise NotImplementedError("not implemented yet")
        # prior_bounds = kwargs["theta_denormalizing"]
        # theta_pred = get_unnormalized_data(theta_pred, prior_bounds)

        # compress_factor = kwargs["compress_factor"]

        # # try:
        # model_rebuild, model_ind_2 = compile_model(
        #     model,
        #     model_pred,
        #     theta_pred,
        #     model_inds,
        #     partition,
        #     verbose=verbose,
        #     max_params=max_params,
        #     constant_names=constant_names,
        #     **kwargs,
        # )
        # x = model_rebuild.run_model()[::compress_factor]

        # if return_model:
        #     return x, model_rebuild
        # else:
        #     return x


def sample_and_run_ddm(
    x_i,
    ddmmodel,
    sampler,
    n=5,
    n_trial=400,
    verbose=False,
    return_last_model=False,
    **kwargs,
):
    """samples and runs n ddms given context x_i

    Args:
        x_i (Tensor): n,2 tensor with corr and err trials. without model inds.
        ddmmodel: ddmmodel class
        sampler (sbi.inference): sampler
        n (int, optional): number of samples. Defaults to 5.
        n_trial (int, optional): number of trials. Defaults to 400.
        verbose (boolean, optional)

    Returns: model_samples, theta_samples, rt_predition
    """

    n_components = len(ddmmodel.theta_partition)
    partition = ddmmodel.theta_partition
    n_params = partition.sum()

    model_samples = torch.zeros((n, n_components))
    theta_samples = torch.zeros((n, n_params))
    rt_predition = torch.zeros((n, n_trial, 2))

    for i in range(n):
        model_sample, theta_sample1 = sampler.sample(1, x_i, verbose=verbose)
        model_samples[i] = model_sample
        if verbose:
            print("model sample:", model_sample)
        theta_samples[i] = inflate_theta(theta_sample1[0], model_sample, partition)

        rt_predition[i], model_compiled = compile_and_run_model(
            ddmmodel,
            model_samples[i],
            theta_samples[i],
            ddmmodel.model_dict.keys(),
            ddmmodel.theta_partition,
            return_model=True,
            verbose=False,
            model_type="ddm",
            n=1,
            n_trial=n_trial,
            **kwargs,
        )

    if return_last_model:
        return model_samples, theta_samples, rt_predition, model_compiled
    else:
        return model_samples, theta_samples, rt_predition


"""
helper functions
"""


def unflatten_theta(theta, partition):
    """puts a flattened theta vector into a structured (n_components, max_params) array,
        inserting nans for non present params

    Args:
        theta (Tensor): (n_theta), flattened theta tensor with nans in non-active components
        partition (_type_): how to split thetas up
    """

    max_params = partition.max()
    n_model_components = len(partition)
    theta_array = torch.zeros(n_model_components, max_params) * torch.nan
    used_params = 0
    for i, item in enumerate(partition):
        theta_array[i, :item] = theta[used_params : used_params + item]
        used_params += item

    return theta_array


def get_model_mask(partition):
    """creates a binary model mask from the partition
    Args:
        partition (_type_): partition of the parameter space
    """

    max_params = partition.max()
    n_model_components = len(partition)

    model_mask = torch.zeros(n_model_components, max_params)
    for i, item in enumerate(partition):
        model_mask[i, :item] = 1

    return model_mask


def inflate_theta(theta, model_ind, partition):
    """puts a parametere prediction into the full parameter space
        insertint nans at the non-present model-components

    Args:
        theta (tensor):  parameter predictions all for the same mask (n,model_ind.sum).
        model_ind (tensor): binary model indices
        partition (tensor): partition of the parameter space

    Returns:
        tensor: inflated model parameters.
    """
    if len(theta.shape) == 1:
        n = 1
    else:
        n = theta.shape[0]

    theta_in = torch.ones(n, partition.sum()) * torch.nan

    model_mask = torch.zeros(n, partition.sum(), dtype=bool)

    count = 0
    for i, item in enumerate(partition):
        if model_ind[i] == 1:
            model_mask[:, count : count + item] = 1
        count += item

    theta_in[model_mask] = theta.flatten()
    return theta_in.squeeze()
