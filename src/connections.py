from typing import Callable, Optional

import flax
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import ones, zeros, constant
from jax import numpy


# Inspiration from @lucidrains for these helper functions
def exists(v):
    return v is not None

def divisible_by(num, den):
    return (num % den) == 0

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def add(x, y):
    return x + y

def get_expand_reduce_stream_function(num_streams, ):

    def expand_reduce(x):
        return jnp.expand_dims(x, -1)
    return expand_reduce

class ResNorm(nn.Module):
    def __init__(self, dim):
        super(ResNorm, self).__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(jnp.zeros(dim))

    def __call__(self, x):
        nn.normalize(x, axis=-1) * self.scale * (self.gamma + 1)

class ResidualConnection(nn.Module):
    def __init__(self, *args, branch: None | nn.Module = None, residual_transformation, **kwargs):
        super(ResidualConnection, self).__init__()
        self.branch = branch
        self.residual_transformation = default(residual_transformation, jax.nn.identity)
        pass
    @nn.compact
    def __call__(self, x):
        return x
    def width_connection(self, residuals):
        return residuals, residuals, dict()

    def depth_connection(self, branch_output, residuals):
        return branch_output + self.residual_transformation(residuals)

    def decorate_branch(self, branch: Callable):



    def residual_transformation(self, branch_output, residuals):


class HyperConnection(nn.Module):
    def __init__(self, dim,
                 nums: int,
                 layer_index: int,
                 dynamic: bool,
                 tanh: bool,
                 scale: bool,
                 channel_first: bool,
                 device=None,
                 dropout: float = 0.0):

        """
        Multi-use case hyperconnection module. Serves different needs from text, vision, video, etc.

        :param dim:
        :param nums:
        :param layer_index:
        :param dynamic:
        :param device:
        :param dropout:
        """
        super(HyperConnection, self).__init__()
        self.dim = dim
        self.nums = nums
        self.layer_index = layer_index
        self.dynamic = dynamic
        self.tanh = tanh
        self.scale = scale
        self.device = device
        self.channel_first = channel_first
        self.dropout = dropout
        self.beta_index: Optional[jnp.ndarray] = None
        self.static_alpha: Optional[jnp.ndarray] = None
        self.dynamic_beta_fn = None
        self.dynamic_alpha_fn = None
        self.static_beta_scale: Optional[jnp.ndarray] = None
        self.static_alpha_scale: Optional[jnp.ndarray] = None
        self.norm = None
        self.static_beta: jnp.ndarray

    def setup(self):
        self.static_beta = self.param('beta_index', ones, (self.nums,))

        init_alpha0 = jnp.zeros((self.nums, 1), device=self.device)
        init_alpha0[(self.layer_index % self.dim), 0] = 1.0



        self.static_alpha = self.param('static_alpha',
                                           lambda k, s: jnp.concatenate([jnp.eye(self.rate), jnp.eye(self.rate)],
                                                                        axis=1),
                                           (self.rate, self.rate * 2))


        if self.dynamic:
            self.dynamic_alpha_fn = nn.Dense(features=self.rate + 1, use_bias=False, kernel_init=nn.initializers.zeros_init())
            self.dynamic_beta_fn = nn.Dense(features=self.dim, use_bias=False, kernel_init=nn.initializers.ones_init())

            if self.use_scale:
                self.static_alpha_scale = self.param('static_alpha_scale', constant(0.01), (1,))
                self.static_beta_scale = self.param('dynamic_beta_scale', constant(0.01), (1,))

            self.norm = nn.LayerNorm(self.dim)

        self.dropout = nn.Dropout(self.dropout)


    def to_streams(self, x):
        if self.channel_first:
            x = x.transpose(-1, 1)

        shape = x.shape

        new_shape = (-1, self.nums)  + shape[1:]

        x = x.reshape(new_shape)

        return x

    def from_streams(self, x):
        shape = x.shape

        new_shape = (shape[0] * shape[1],) + shape[2:]

        x = x.reshape(new_shape)

        if self.channel_first:
            x = x.transpose(-1, 1)

        return x

    def dept_connection(self, x):
        """

        :param x: residual
        :return: branch_input, mix_h, beta
        """




    def width_connection(self, x):
        """
        :param x: residual
        :return: branch_input, mix_h, beta
        """

        x = self.to_streams(x)

        if self.dynamic:

            normed = self.norm(x)

            dynamic_alpha = self.dynamic_alpha_fn(normed)

            if self.tanh:
                dynamic_alpha = jnp.tanh(dynamic_alpha)

            if self.use_scale:
                dynamic_beta = self.dynamic_beta_fn(normed)

            num_spatial_dim = jnp.ndim(x) - 3

            static_alpha = self.static_alpha.reshape(1, self.nums, *([1] * num_spatial_dim), self.num_streams + 1)

            alpha = dynamic_alpha + static_alpha

            dynamic_beta = self.dynamic_beta_fn(normed).squeeze(-1)

            if self.tanh:
                dynamic_beta = jnp.tanh(dynamic_beta)
            if self.use_scale:
                dynamic_beta = dynamic_beta + self.beta_scale


            static_beta = self.static_beta.reshape(1, self.num_streams, *([1] * num_spatial_dim))

            beta = static_beta  + dynamic_beta

        else:

            num_spatial_dim = jnp.ndim(x) - 3

            alpha = self.static_alpha.reshape(1, self.nums, *([1] * num_spatial_dim), self.num_streams + 1)

            beta = self.static_beta.reshape(1, self.nums, *([1] * num_spatial_dim), self.num_streams + 1)


        mix_h = jnp.einsum('bs...t,bs...d->bt...d', alpha, h)

        branch_input = mix_h[:, 0]

        residuals = mix_h[:, 1:]

        if self.channel_first:
            branch_input = jnp.moveaxis(a=branch_input, source=1, destination=-1)

        return branch_input, residuals, beta










        pass


    def __class__(self):
        pass



    @nn.compact
    def __call__(self, x):
        pass

    def


class FracConnection(nn.Module):
    def __init__(self, dim, rate, config, dynamic_beta, dynamic_alpha, device=None):
        super(FracConnection, self).__init__()
        self.norm = None
        self.static_alpha = None
        self.static_beta = None
        self.dynamic_alpha_fn = None
        self.dynamic_beta_fn = None
        self.static_alpha_scale = None
        self.dynamic_beta_scale = None
        self.dim = dim
        self.rate = rate
        self.dynamic_beta = dynamic_beta
        self.dynamic_alpha = dynamic_alpha
        self.tanh = config.use_tahn
        self.hc_norm = config.hc_norm
        self.use_scale = config.use_scale

    def setup(self):
            self.static_beta = self.param('static_beta', ones, (self.rate,))
            self.static_alpha = self.param('static_alpha',
                                           lambda k, s: jnp.concatenate([jnp.eye(self.rate), jnp.eye(self.rate)],
                                                                        axis=1),
                                           (self.rate, self.rate * 2))

            if self.dynamic_alpha:
                self.dynamic_alpha_fn = self.param("dynamic_alpha_fn", zeros, (self.dim // self.rate, self.rate*2))

            if self.dynamic_beta:
                self.dynamic_beta_fn = self.param("dynamic_beta_fn", ones, (self.dim // self.rate,))

            if self.use_scale:
                self.static_alpha_scale = self.param('static_alpha_scale', constant(0.01), (1,))
                self.dynamic_beta_scale = self.param('dynamic_beta_scale', constant(0.01), (1,))

            if self.hc_norm:
                self.norm = ResNorm(self.dim // self.rate)


    def width_connection(self, m: jax.numpy.ndarray):
        """
        :param m: (batch, seq, dim)
        :return: Branch Input, Residual, Beta
        """
        m_shape = m.shape

        frac_dim = m_shape[-1] // self.rate

        m_reshape = m.reshape(m_shape[:-1] + (self.rate, frac_dim))

        if self.hc_norm:
            norm = (self.norm(m_reshape))
        else:
            norm = m_reshape

        # alpha
        if self.dynamic_alpha:
            dynamic_alpha = norm @ self.dynamic_alpha_fn
            if self.tanh:
                dynamic_alpha = jnp.tanh(norm @ self.dynamic_alpha_fn)
            dynamic_alpha = dynamic_alpha * self.dynamic_alpha_scale
            alpha = dynamic_alpha + self.static_alpha
        else:
            alpha = self.static_alpha

        # beta
        if self.dynamic_beta:
            dynamic_beta = norm @ self.dynamic_beta_fn
            if self.tanh:
                dynamic_beta = jnp.tanh(norm @ self.dynamic_beta_fn)
            dynamic_beta = dynamic_beta * self.dynamic_beta_scale
            beta = dynamic_beta + self.static_beta
        else:
            beta = self.static_beta

        mix_h = alpha.swapaxes(-1, -2) @ m_reshape

        branch_input = mix_h[..., :self.rate, :]
        residuals = mix_h[..., self.rate:, :]

        branch_input = branch_input.reshape(m_shape)  # (batch, seq, dim)

        return branch_input, residuals, beta

    def dept_connection(self, residuals: jnp.ndarray, beta: jnp.ndarray, h_o: jnp.ndarray):
        """
        :param residuals: (..., rate, frac_dim)
        :param beta: (..., rate)
        :param h_o: (batch, seq, frac_dim)
        :return: dept connection which is (batch, seq, dim)
        """
        h_o_shape = h_o.shape

        frac_dim = h_o_shape[-1] // self.rate

        h_o_reshape = h_o.reshape(h_o_shape[:-1] + (self.rate, frac_dim))

        weighted_output = beta[..., :, None] * h_o_reshape

        h = weighted_output + residuals

        return h.reshape(h_o_shape)

    @nn.compact
    def __call__(self, x):
        pass