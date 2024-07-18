import numpy as np

"""
define class for model components
"""


class submodel:
    """
    container for information on model components of a submodel
    """

    def __init__(self, model_dict, prior, prior_of_constants, p=None, unique=False):
        self.model_dict = model_dict
        self.prior = prior
        self.unique = unique  # if every component can be sampled more than once
        self.constants = self.get_constants()
        self.prior_of_constants = prior_of_constants  # dict of sampling_fcts
        assert len(self.constants) == len(self.prior_of_constants)

    def sample(self):
        if self.prior == "uniform":
            s = np.random.choice(list(self.model_dict.keys()))
        elif self.prior == "categorical":
            s = np.random.choice(list(self.model_dict.keys()), p=p)
        return s

    def sample_constants(self):
        samples = dict.fromkeys(self.prior_of_constants.keys())
        for key, item in self.prior_of_constants.items():
            samples[key] = dict.fromkeys(self.prior_of_constants[key].keys())
            for key2, distribution in item.items():
                samples[key][key2] = distribution()

        return samples

    def get_constants(self, const_name="c_", n_digits=3):
        """
        extracts constants for model component
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


class model:
    """
    base class for different models
    """

    def __init__(self, model_components, variables, n_per_comp):
        self.model_components = model_components  # list of submodels
        self.variables = variables
        self.n_per_comp = n_per_comp

        assert len(self.n_per_comp) == len(self.model_components)

    def sample_model_skeleton(self):
        raise Exception("Not Implemented")

    def sample_constants(self, model_ind):
        raise Exception("Not Implemented")

    def compile_model(self, model_ind, constants):
        raise Exception("Not Implemented")

    def sample_model(self):
        model_ind = self.sample_model_skeleton()
        constants = self.sample_constants(model_ind)
        model = self.compile_model(model_ind, constants)

        return model
