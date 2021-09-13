import torch
import numpy as np
from collections import OrderedDict

from model_transformer.model_utils import last_dims_tuple
from my_common.my_helper import is_ndarray, is_tensor
from util.jp_util import JpUtil
from util.constants import Constants, InputType, Regularization, RegRepresentation, TaskType
from common.helper import print_stats, Util
from common.torch_util import TorchUtil as tu


class FiniteDifferences:
    @staticmethod
    def make_d_step_center(eps, seqlen):
        """
        Create a doubly stochastic matrix by a
        convex combination between identity matrix
        and the matrix 1/n * ones() (this matrix computes a mean)

        :param eps: The desired norm between output and identity matrix
        :param seqlen: The dimension of the matrix
        """
        divisor = float(np.sqrt(seqlen - 1))
        assert divisor > 0 and eps < divisor  # eps must be small enough to get a valid convex comb
        c = 1 - eps / divisor
        centroid = torch.full((seqlen, seqlen), 1. / seqlen)
        return c * torch.eye(seqlen) + (1 - c) * centroid

    @staticmethod
    def make_d_step_edge(eps, seqlen, random_segment=False, permute_range=None):
        """
        Create a doubly stochastic matrix by a
        convex combination between identity matrix
        and the matrix with two rows swapped (a neighbor to the identity on the Birkhoff polytope)

        :param eps: The desired norm between output and identity matrix
        :param seqlen: The dimension of the matrix
        :param random_segment: If true, randomly pick two rows; otherwise rows must be consecutive
        :param permute_range: The range of the rows to be swapped
        """
        assert 0. < eps < 2., "Epsilon must be between 0 and 2"
        c = 1. - eps / 2.
        if permute_range is None:
            permute_range = seqlen
        else:
            assert isinstance(permute_range, int) and 1 < permute_range <= seqlen

        assert isinstance(random_segment, bool)
        if random_segment:
            first_idx, second_idx = np.random.choice(permute_range, 2, replace=False).tolist()
        else:
            first_idx = torch.randint(permute_range - 1, (1,)).item()
            second_idx = (first_idx + 1)

        D = torch.eye(seqlen)
        D[first_idx, second_idx] = 1. - c
        D[second_idx, first_idx] = 1. - c
        D[first_idx, first_idx] = c
        D[second_idx, second_idx] = c

        return D

    @staticmethod
    def make_d_step_edges(eps, seqlen, random_segment=False, permute_range=None, n_vert_list=None):
        """
        """
        if n_vert_list is None:
            return FiniteDifferences.make_d_step_edge(eps, seqlen, random_segment, permute_range)
        else:
            if isinstance(n_vert_list, list):
                assert len(n_vert_list) > 0
            elif is_ndarray(n_vert_list) or is_tensor(n_vert_list):
                assert n_vert_list.ndim == 1 and len(n_vert_list) > 0
                n_vert_list = n_vert_list.tolist()
            else:
                raise TypeError(f"Expected n_vert_list to be a list, ndarray or tensor, but got {type(n_vert_list)}.")
            return torch.stack(Util.lm(lambda n: FiniteDifferences.make_d_step_edge(eps, seqlen, random_segment, n), n_vert_list))

    @staticmethod
    def make_d_step_vertex(seqlen, permute_range=None):
        """
        Create a permutation matrix by uniformly sampling from the vertices of the Birkhoff polytope

        :param seqlen: The dimension of the matrix
        :param permute_range: The range of the rows to be permuted
        """
        if permute_range is None:
            permute_range = seqlen
        else:
            assert 1 < permute_range <= seqlen

        D = torch.eye(seqlen)
        perm = torch.randperm(permute_range)
        D[:permute_range] = D[perm]

        return D

class JpRegularizer:
    def __init__(self, regularization, input_type, total_permutations=None, strength=None, eps=None, representations=False,
                 normed = False, power=2, tangent_prop=False, tp_config=None, fixed_edge=False, random_segment=False,
                 permute_positional_encoding=False, task_type=None, agg_penalty_by_node=False):
        """
        :param total_permutations: Number of permutations to use for
        computing the penalty, >= number used for training

        :param strength: regularization strength (if any)
        :param eps: -- size of finite difference for the diff-step method
        :param representations: -- representation of which regularization is taken to penalize the permutation sensitivity
        :param normed and power: -- The default values correspond to the Tangent Prop paper, squared L2 norm.
        :param tangent prop: -- If true, we are using Tangent Prop style penalty that first estimate tangent vectors by
                                taking diff step on x and then calculate the Jacobian - tangent vector product
        :param tp_config: -- configurations of TangentProp, including
                             tp_num_tangent_vectors -- Number of tangent vectors used for TangentProp (default: 1)
        :param fixed_edge: -- Fix the edge of DS matrices for diff_step_edge regularization (to the one between 1st and 2nd perms)
        :param random_segment: -- If true, randomly swap two rows for the edge-step DS matrices; otherwise rows must be consecutive
        :param permute_positional_encoding: -- If true, multiply DS matrices with positional encodings instead of input data
        :param task_type: task_type for determining desired properties, e.g., permutation-equivariance on node classification task
        :param agg_penalty_by_node: -- If true, mean aggregate penalty by nodes; otherwise by graphs (with normalization by graph size)
        """
        assert regularization in Regularization
        assert input_type in InputType
        assert representations in RegRepresentation

        for bool_var in [normed, tangent_prop, fixed_edge, random_segment, permute_positional_encoding, agg_penalty_by_node]:
            assert isinstance(bool_var, bool), \
                f"JpRegularizer expected a bool but got type {type(bool_var)} for some input"

        assert isinstance(power, int) and power > 0

        if total_permutations is None:
            assert regularization in (Regularization.none, Regularization.diff_step_center,
                                      Regularization.diff_step_edge, Regularization.diff_step_basis,
                                      Regularization.diff_step_naive)
        else:
            assert isinstance(total_permutations, int)
            if regularization == Regularization.pw_diff:
                assert total_permutations > 1, "pw_diff regularizer needs at least 2 permutations"
            else:
                assert total_permutations > 0

        if strength is None:
            assert regularization == Regularization.none
        else:
            assert isinstance(strength, float) and strength >= 0.0

        if regularization in (Regularization.diff_step_center, Regularization.diff_step_edge,
                              Regularization.diff_step_basis, Regularization.diff_step_naive):
            assert representations != RegRepresentation.none
            if regularization == Regularization.diff_step_naive:
                assert eps is None or (isinstance(eps, float) and eps == 0.0)
            else:
                assert eps is not None and isinstance(eps, float) and eps > 0.0
        else:
            assert tangent_prop is False, "tangent_prop can only be true for a step-type regularization"

        if fixed_edge or random_segment:
            assert regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis), \
                "Fixed edge was specified, but edge-step regularization is not in effect."
            assert not (fixed_edge and random_segment), "Fixed-edge and random-edge cannot be in effect simultaneously."

        if tangent_prop:
            assert isinstance(tp_config, OrderedDict)
            assert all(config in tp_config for config in ['tp_num_tangent_vectors'])
            assert isinstance(tp_config['tp_num_tangent_vectors'], int) and tp_config['tp_num_tangent_vectors'] > 0
            if tp_config['tp_num_tangent_vectors'] > 1:
                assert not fixed_edge, "Edge must not be fixed for More-tangent-vector TangentProp."
                assert regularization in (Regularization.edge, Regularization.basis, Regularization.naive), \
                    "More-tangent-vector TangentProp was specified, but non-center regularization is not in effect."
            self.tp_num_tangent_vectors = tp_config['tp_num_tangent_vectors']

        if task_type is not None:
            assert isinstance(task_type, TaskType)
        if agg_penalty_by_node:
            assert task_type == TaskType.node_classification, "Agg_penalty_by_node is True (aggregating penalty by nodes), " \
                                                              "but the task_type is not node_classification."

        self.input_type = input_type
        self.representations = representations
        self.regularization = regularization
        self.total_permutations = total_permutations
        self.strength = strength
        self.eps = eps
        self.tangent_prop = tangent_prop
        self.permute_positional_encoding = permute_positional_encoding
        self.normed = normed
        self.power = power

        self.D = None  # D is fixed in center step; retained to save time
        # permute range of edge-step D
        self.edge_permute_range = 2 if fixed_edge else None # specified for fixed edge or BA experiments or variable-size graphs
        self.random_segment = random_segment

        self.task_type = task_type
        self.agg_penalty_by_node = agg_penalty_by_node

    def __str__(self):
        vars_to_print = Util.dictize(regularization=self.regularization.name,
                                     input_type=self.input_type.name,
                                     strength=self.strength,
                                     eps=self.eps,
                                     representations=self.representations,
                                     power=self.power,
                                     normed=self.normed,
                                     tangent_prop=self.tangent_prop,
                                     permute_positional_encoding=self.permute_positional_encoding,
                                     edge_permute_range=self.edge_permute_range,
                                     random_segment = self.random_segment,
                                     task_type=self.task_type,
                                     agg_penalty_by_node=self.agg_penalty_by_node)

        if self.tangent_prop:
            vars_to_print.update(Util.dictize(
                tp_num_tangent_vectors=self.tp_num_tangent_vectors
            ))

        _str = "JpRegularizer:\n "
        for kk, vv in vars_to_print.items():
            _str += f"{kk}: {vv}\n "
        return _str

    def __call__(self, representations, model, disable_gradient, adjmats=None,
                 sets=None, positional_encodings=None, vertex_features=None, n_vert_list=None):
        """
        :param representations: tensor of shape (num_sequences, num_permutations, dimension)
        """
        # FIXME: Edge-step / Random-segment / Naive Doubly-stochastic matrices depend on the batch size and BATCH LIMIT!
        #        For every new batch, either for mini-batching or computational tractability, a new Dst will be generated,
        #        which leads to unexpected dependency of penalties on batch_limit.
        assert representations.dim() in (2, 3)
        assert isinstance(disable_gradient, bool)

        if self.input_type == InputType.graph:
            if self.permute_positional_encoding:
                assert positional_encodings is not None
                # Assume same graph sizes when positional encoding is in use
                if positional_encodings.dim() == 3:
                    # Onehot ID's or positional wmbeddings w/ variable num of vertices
                    seq_len = positional_encodings.shape[1]
                elif positional_encodings.dim() == 2:
                    seq_len = positional_encodings.shape[0]  # Positional Embeddings w/o n_vert_list
                else:
                    raise ValueError(f"Unexpected dim of positional_encodings: {positional_encodings.ndim()}")
                if not self.tangent_prop:  # both adjmats and positional_encodings are required to compute finite diff
                    assert adjmats is not None
                    if positional_encodings.dim() == 3:
                        assert positional_encodings.shape[0] == adjmats.shape[0] and positional_encodings.shape[1] == adjmats.shape[-1]
                    else:
                        assert positional_encodings.shape[0] == adjmats.shape[1]
            else:
                assert adjmats is not None  # adjmats are always required to compute finite diff
                seq_len = adjmats[0].shape[-1]  # Currently assume same sizes (simplicity)

        else:  # sets
            if self.permute_positional_encoding:
                assert positional_encodings is not None and positional_encodings.dim() == 2
                seq_len = positional_encodings.shape[0]  # Assume same set sizes when positional encoding is in use
                if not self.tangent_prop:  # both sets and positional_encodings are required to compute finite diff
                    assert sets is not None and positional_encodings.shape == sets[0].shape
            else:
                assert sets is not None
                seq_len = sets[0].shape[0]  # Currently assume same sizes (simplicity)

        previous_grad_state = torch.is_grad_enabled()
        if disable_gradient:
            torch.set_grad_enabled(False)

        if self.regularization in (Regularization.diff_step_center,
                                   Regularization.diff_step_edge,
                                   Regularization.diff_step_basis,
                                   Regularization.diff_step_naive):

            # If computing the "basis" penalty, compute both the center-step and edge-step penalties
            # o.w. just compute one
            if self.regularization in (Regularization.diff_step_center, Regularization.diff_step_basis):
                # Finite diff regularization, move input to the very center of permuto
                # (we construct the relevant matrix here which is needed for the center step)
                if self.D is None:
                    self.D = tu.move(FiniteDifferences.make_d_step_center(self.eps, seq_len))
                if self.tangent_prop:
                    center_penalty = self.diff_step_tangent_prop(representations, self.D,
                                                                 adjmats, sets, positional_encodings, vertex_features)
                else:
                    center_penalty = self.diff_step(representations, model, self.D, adjmats, sets, positional_encodings, vertex_features)

            if self.regularization in (Regularization.diff_step_edge, Regularization.diff_step_basis, Regularization.naive):
                # Finite diff regularization, move input along an edge
                # Note, if there are duplicates in the input sequence, the edge penalty is zero,
                # but it will be unlikely and it's not worth checking here.
                if self.tangent_prop:
                    # To prevent numerical overflow:
                    # Don't sum then divide, divide as we go
                    divisor = 1. / self.tp_num_tangent_vectors
                    if self.representations == RegRepresentation.pe:
                        edge_penalty = tu.move(torch.zeros((1,)))  # no batch size in positional embedding (repr) penalty
                    else:
                        edge_penalty = tu.move(torch.zeros(representations.shape[0]))
                    for j in range(self.tp_num_tangent_vectors):
                        if self.regularization != Regularization.naive:
                            D = tu.move(FiniteDifferences.make_d_step_edges(self.eps, seq_len, self.random_segment, self.edge_permute_range, n_vert_list))
                        else:
                            D = tu.move(FiniteDifferences.make_d_step_vertex(seq_len, self.edge_permute_range))
                        edge_penalty += divisor * self.diff_step_tangent_prop(representations, D,
                                                                              adjmats, sets, positional_encodings, vertex_features)
                else:
                    if self.regularization != Regularization.naive:
                        D = tu.move(FiniteDifferences.make_d_step_edges(self.eps, seq_len, self.random_segment, self.edge_permute_range, n_vert_list))
                    else:
                        D = tu.move(FiniteDifferences.make_d_step_vertex(seq_len, self.edge_permute_range))
                    edge_penalty = self.diff_step(representations, model, D, adjmats, sets, positional_encodings, vertex_features)

            if self.regularization == Regularization.diff_step_basis:
                # Assign equal weights to center_penalty and edge_penalty
                result = self.strength * 0.5 * (center_penalty + edge_penalty)
            elif self.regularization == Regularization.diff_step_center:
                result = self.strength * center_penalty
            elif self.regularization == Regularization.diff_step_edge or Regularization.diff_step_naive:
                result = self.strength * edge_penalty

        # Implementing "no regularization" for completeness and flexibility. Not recommended to use this way.
        elif self.regularization == Regularization.none:
            result = torch.zeros_like(representations)
        else:
            raise NotImplementedError(f"Regularization not implemented in __call__: {self.regularization}")

        torch.set_grad_enabled(previous_grad_state)
        return result

    def __unify_shapes(self, in_tensor):
        """
        Unify the shapes to be two-dimensional
        Shape depends on whether we're regularizing the prediction or the latent
        Sometimes, representations will have an additional dimension
        because other regularizers require it...the unsqueeze() is in loss.
        """
        if in_tensor.dim() == 2:
            return in_tensor
        elif in_tensor.dim() == 3 and in_tensor.shape[1] == 1:
            return in_tensor.squeeze(1)
        elif in_tensor.dim() == 1:
            return in_tensor.unsqueeze(1)
        else:
            raise RuntimeError("Unexpected dimension based on implementation")

    def _compute_penalty(self, raw_penalty):

        # Compute last_dims
        if self.representations == RegRepresentation.pe:
            # Dim of positional embedding is (seq_len, dim_pe) regardless of batch size
            last_dims = last_dims_tuple(raw_penalty, 0)
        elif self.task_type == TaskType.node_classification:
            # For node classification, we aggregate the penalty by graph/node
            assert raw_penalty.ndim == 3
            last_dims = last_dims_tuple(raw_penalty, 2 if self.agg_penalty_by_node else 1)
        else:
            last_dims = last_dims_tuple(raw_penalty, 1)

        # Compute penalty
        if self.normed:
            penalty = torch.norm(raw_penalty, p=self.power, dim=last_dims)
        elif self.power == 1:
            penalty = torch.sum(torch.abs(raw_penalty), dim=last_dims)
        else:
            if self.power % 2 != 0:
                raw_penalty = torch.abs(raw_penalty)
            penalty = torch.sum(torch.pow(raw_penalty, self.power), dim=last_dims)

        # Aggregate penalty
        if self.task_type == TaskType.node_classification:
            if self.agg_penalty_by_node:
                # Take the average penalty of nodes for each graph
                penalty = penalty.mean(dim=1)
            else:
                # Normalize the penalty by graph size (# nodes) to be consistent with loss
                penalty = torch.div(penalty, raw_penalty.shape[1])

        return penalty

    def diff_step(self, representations, model, D, adjmats, sets, positional_encodings, vertex_features):
        """
        Finite difference gradient estimator
        Uses D that move input along permutohedron
        """

        # Send sequence along direction of its permutohedron
        if self.input_type == InputType.graph:
            if not self.permute_positional_encoding:
                adjmats_inside = D @ adjmats @ D.transpose(0, 1)
                vertex_features_inside = D @ vertex_features if vertex_features is not None else None
                predictions_inside, latent_inside, pe_inside = model(adjmats_inside, positional_encodings, vertex_features_inside)
            else:
                positional_encodings_inside = D @ positional_encodings
                predictions_inside, latent_inside, pe_inside = model(adjmats, positional_encodings_inside, vertex_features)
        else:  # sets
            # TODO: add the option of fixing word embeddings during regularization pass (for backward())
            if self.permute_positional_encoding:
                positional_encodings_inside = D @ positional_encodings
                predictions_inside, latent_inside, _, pe_inside = model(sets, positional_encodings_inside)
            else:
                sets_inside = D @ sets
                predictions_inside, latent_inside, _, pe_inside = model(sets_inside)

        if self.representations == RegRepresentation.prediction:
            representations_inside = predictions_inside
        elif self.representations == RegRepresentation.latent:
            representations_inside = latent_inside
        elif self.representations == RegRepresentation.positional_embedding:
            representations_inside = pe_inside
        else:
            raise NotImplementedError(f"Haven't implemented regularization for {self.representations.name} yet.")

        # For node classification, models should be regularized towards permutation-equivariance if we permute adjmats
        if self.task_type == TaskType.node_classification and not self.permute_positional_encoding \
                and self.representations in (RegRepresentation.prediction, RegRepresentation.latent):
            representations_inside = D @ representations_inside

        if representations.shape != representations_inside.shape:
            print(f"representations.shape is {representations.shape}\nrepresentations_inside.shape is {representations_inside.shape}")
            raise RuntimeError("shape mismatch")

        if self.regularization == Regularization.naive:
            eps_divisor = 1.
        else:
            eps_divisor = self.eps if self.normed else pow(self.eps, self.power)

        return torch.div(self._compute_penalty(representations - representations_inside), eps_divisor)

    def diff_step_tangent_prop(self, representations, D, adjmats, sets, positional_encodings, vertex_features):
        """
        Finite difference gradient estimator (tangent prop)
        Uses D that move input along permutohedron
        """
        if self.permute_positional_encoding:
            input_obj = positional_encodings
            assert input_obj.dim() in (2, 3)
        elif self.input_type == InputType.graph:
            if vertex_features is None:
                input_obj = adjmats
                assert input_obj.dim() == 3
            else:
                input_obj = (adjmats, vertex_features)
                assert input_obj[0].dim() == 3 and input_obj[1].dim() == 3
                raise NotImplementedError("Haven't implemented TangentProp for graphs with vertex features.")
        else:
            input_obj = sets
            assert input_obj.dim() == 3

        assert tu.all_requires_grad(input_obj), "Tangent prop regularizer requires grad for input_obj, see code."

        # Send sequence along direction of its permutohedron
        if self.permute_positional_encoding or self.input_type == InputType.set:
            input_inside = D @ input_obj
        else:  # graphs, permuting adjmats; assume featureless for now
            if vertex_features is None:
                input_inside = D @ input_obj @ D.transpose(0, 1)
            else:
                raise NotImplementedError("Haven't implemented TangentProp for graphs with vertex features.")

        # Compute Jacobian-vector product with tangent vector
        if self.regularization == Regularization.naive:
            tangent_vec = input_obj - input_inside
        else:
            tangent_vec = (input_obj - input_inside) / self.eps

        # For node classification, models should regularized towards permutation-equivariance if we permute adjmats
        if self.task_type == TaskType.node_classification and not self.permute_positional_encoding \
                and self.representations in (RegRepresentation.prediction, RegRepresentation.latent):
            representations_for_jvp = D @ representations
        else:
            representations_for_jvp = representations
        jvp = self.get_jvp(representations_for_jvp, input_obj, tangent_vec)

        # Compute penalty
        return self._compute_penalty(jvp)

    @staticmethod
    def get_jvp(y, x, vec):
        """
        Construct a differentiable Jacobian-vector product for a function
        Adapted from: https://gist.github.com/ybj14/7738b119768af2fe765a2d63688f5496
        Trick from: https://j-towns.github.io/2017/06/12/A-new-trick.html

        :param y: output of a function
        :param x: input of a function
        :param vec: vector
        """
        u = torch.zeros_like(y, requires_grad=True)  # u is an auxiliary variable and could be arbitary
        ujp = torch.autograd.grad(y, x, grad_outputs=u, create_graph=True)[0]
        jvp = torch.autograd.grad(ujp, u,
                                  grad_outputs=vec,
                                  create_graph=True)[0]
        return jvp

    def perm_grad_penalty(self):
        """Gradient of f-arrow at a particular function"""
        raise NotImplementedError


# Test and debug
if __name__ == "__main__":
    tp_config = OrderedDict({'tp_power': 3, 'tp_normed': True, 'tp_num_tangent_vectors': 5})
    jpr = JpRegularizer(regularization=Regularization.diff_step_edge, input_type=InputType.set,
                        strength=123., eps=.3211, representations=RegRepresentation.pred,
                        tangent_prop=True,
                        tp_config=tp_config)
    print(jpr)

