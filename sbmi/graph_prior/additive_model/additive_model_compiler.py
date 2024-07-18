import numpy as np
import sympy as sp
from sympy import tanh, Piecewise

from sbmi.graph_prior.graph_model import Model

# from model_def import model


######################
# Using graph layout
######################


class AdditiveModelGraph(Model):
    def __init__(
        self,
        model_dict,
        unique_components,
        node_priors,
        edges,
        variables,
        additive_model_ids,
        CompileClass,
        node_prior_bounds,
        theta_partition,
        pos_coupling=None,
        neg_coupling=None,
        exclusive_nodes=None,
        pos_coupling_factor=2.0,
        dt=0.01,
    ):
        self.additive_model_ids = additive_model_ids
        self.dt = dt
        super().__init__(
            model_dict,
            unique_components,
            node_priors,
            edges,
            variables,
            CompileClass,
            node_prior_bounds,
            theta_partition,
            pos_coupling=pos_coupling,
            neg_coupling=neg_coupling,
            exclusive_nodes=exclusive_nodes,
            pos_coupling_factor=pos_coupling_factor,
        )

    def compile_model(self, model_ind, constants):
        """ """

        # read in sympy variables
        sp_variable = [sp.var(item) for item in self.variables]

        # list additive expressions
        additive_expressions = []
        n_a = -1
        for i, comp_id in enumerate(model_ind):
            if comp_id in self.additive_model_ids:
                additive_expressions.append(sp.parse_expr(self.model_dict[comp_id]))
                n_a += 1
                for constant in constants[i]:
                    additive_expressions[n_a] = additive_expressions[n_a].replace(
                        sp.parse_expr(constant), constants[i][constant]
                    )

        formula = 0
        for item in additive_expressions:
            formula += item

        rhs = sp.lambdify(sp_variable, formula)

        model_to_run = self.CompileClass(rhs, formula, model_ind, constants, dt=self.dt)

        return model_to_run


class CompiledAdditiveModel:
    def __init__(self, model_func, formula, model_ind, constants, dt):
        self.model_func = model_func
        self.formula = formula
        self.model_ind = model_ind  # indeces of the used model components
        self.constants = constants
        self.dt = dt

    def run_model(self, n=1, return_t=True, *args):
        """
        run the model n times
        Args:
            n (int, optional): number of model runs. Defaults to 1.

        Returns:
            tuple or two arrayz: t,x
        """

        # define t
        t = np.arange(0, 10, self.dt)
        sol_all = []
        for i in range(n):
            offset = np.zeros(len(t))
            if 20 in self.model_ind:
                offset += (
                    np.ones(len(t))
                    * np.array(self.constants)[self.model_ind == 20][0]["c_1"]
                )
            if 21 in self.model_ind:
                offset += (
                    np.ones(len(t))
                    * np.array(self.constants)[self.model_ind == 21][0]["c_1"]
                )

            # noise on initial condition
            if 101 in self.model_ind:
                initial_condition = np.random.normal(
                    np.array(self.constants)[self.model_ind == 101][0]["c_1"]
                )
            else:
                initial_condition = 0

            # run model
            sol = self.model_func(t) + initial_condition + offset

            # add observation noise
            if 100 in self.model_ind:
                # independent whit noise
                noise = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 100][0]["c_1"],
                    len(sol),
                )
                sol = sol + noise

            if 102 in self.model_ind:
                # correlated noise
                kernel = GaussKernel(0.05)  # 0.03 in v5, 0.05 in v6 #0.02
                noise_raw = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 102][0]["c_1"],
                    len(sol),
                )

                noise = kernel.convolve(noise_raw)

                sol = sol + noise

            if 103 in self.model_ind:
                # increasing noise over time
                noise = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 103][0]["c_1"],
                    len(sol),
                )
                # sol = sol + noise * t * 0.5
                sol = sol + noise * (t + 1)

            if 104 in self.model_ind:
                # decreasing noise over time
                noise = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 104][0]["c_1"],
                    len(sol),
                )
                sol = sol + noise * (11 - t)

            if 105 in self.model_ind:
                # increasing quadratic noise over time
                noise = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 105][0]["c_1"],
                    len(sol),
                )
                sol = sol + noise * (t**2 + 1)

            if 106 in self.model_ind:
                # decreasing quadratic noise over time
                noise = np.random.normal(
                    0,
                    np.array(self.constants)[self.model_ind == 106][0]["c_1"],
                    len(sol),
                )
                sol = sol + noise * (11 - t**2)

            sol_all.append(sol)

        if return_t:
            return t, np.array(sol_all)
        else:
            return np.array(sol_all)
