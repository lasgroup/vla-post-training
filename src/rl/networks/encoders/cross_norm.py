"""Normalization modules for Flax."""

from typing import (Any, Callable, Optional, Tuple, Iterable, Union)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp



PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?

Axes = Union[int, Iterable[int]]

import flax.nnx as nnx

def _canonicalize_axes(rank: int, axes: Axes) -> Tuple[int, ...]:
  """Returns a tuple of deduplicated, sorted, and positive axes."""
  if not isinstance(axes, Iterable):
    axes = (axes,)
  return tuple(set([rank + axis if axis < 0 else axis for axis in axes]))


def _abs_sq(x):
  """Computes the elementwise square of the absolute value |x|^2."""
  if jnp.iscomplexobj(x):
    return lax.square(lax.real(x)) + lax.square(lax.imag(x))
  else:
    return lax.square(x)


def _compute_stats(x: Array, axes: Axes,
                   axis_name: Optional[str] = None,
                   axis_index_groups: Any = None,
                   alpha: float = 0.5):
  """Computes mean and variance statistics.
  This implementation takes care of a few important details:
  - Computes in float32 precision for half precision inputs
  -  mean and variance is computable in a single XLA fusion,
    by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
  - Clips negative variances to zero which can happen due to
    roundoff errors. This avoids downstream NaNs.
  - Supports averaging across a parallel axis and subgroups of a parallel axis
    with a single `lax.pmean` call to avoid latency.
  Arguments:
    x: Input array.
    axes: The axes in ``x`` to compute mean and variance statistics for.
    axis_name: Optional name for the pmapped axis to compute mean over.
    axis_index_groups: Optional axis indices.
  Returns:
    A pair ``(mean, var)``.
  """
  # promote x to at least float32, this avoids half precision computation
  # but preserves double or complex floating points
  x = jnp.asarray(x, jnp.promote_types(jnp.float32, jnp.result_type(x)))
  split_1, split_2 = jnp.split(x, 2, axis=0) # split x into two parts
  mean_s1 = jnp.mean(split_1, axes)
  mean_s2 = jnp.mean(split_2, axes)
  
  mean2_s1 = jnp.mean(_abs_sq(split_1), axes)
  mean2_s2 = jnp.mean(_abs_sq(split_2), axes)

  mean = alpha * mean_s1 + (1 - alpha) * mean_s2

  if axis_name is not None:
    concatenated_mean = jnp.concatenate([mean, mean2])
    mean, mean2 = jnp.split(
        lax.pmean(
            concatenated_mean,
            axis_name=axis_name,
            axis_index_groups=axis_index_groups), 2)
  # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
  # to floating point round-off errors.
  var_s1 = mean2_s1 - _abs_sq(mean_s1)
  var_s2 = mean2_s2 - _abs_sq(mean_s2)
  var = alpha * var_s1 + (1 - alpha) * var_s2

  var = jnp.maximum(0., var)
  return mean, var


def _normalize(x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: Dtype, param_dtype: Dtype,
               epsilon: float,
               scale: Optional[Array],
               bias: Optional[Array]):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
  Arguments:
  Arguments:
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: Dtype of the returned result.
    param_dtype: Dtype of the parameters.
    epsilon: Normalization epsilon.
    scale: Scale parameter (optional).
    bias: Bias parameter (optional).
  Returns:
    The normalized input.
  """
  reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = _canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  mul = lax.rsqrt(var + epsilon)
  if scale is not None:
    mul *= scale.reshape(feature_shape)
  y *= mul
  if bias is not None:
    y += bias.reshape(feature_shape)
  return jnp.asarray(y, dtype)

class CrossNorm(nnx.Module):
  """CrossNorm Module.

  Attributes:
    use_running_average: if True, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-spatial axis.
      The mean and variance are calculated at the non-feature axes.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    use_bias:  if True, adding a bias term to the output.
    use_scale: if True, scaling the output.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None).
    alpha: interpolation factor for CrossNorm statistics.
  """
  def __init__(self,
               x_example: Union[Array, int],
               use_running_average: Optional[bool] = None,
               axis: int = -1,
               momentum: float = 0.99,
               epsilon: float = 1e-5,
               dtype: Dtype = jnp.float32,
               param_dtype: Dtype = jnp.float32,
               use_bias: bool = True,
               use_scale: bool = True,
               bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros,
               scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones,
               axis_name: Optional[str] = None,
               axis_index_groups: Any = None,
               alpha: float = 0.5,
               *, rngs: nnx.Rngs):
    self.use_running_average = use_running_average
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon
    self.dtype = dtype
    self.param_dtype = param_dtype
    self.use_bias = use_bias
    self.use_scale = use_scale
    self.axis_name = axis_name
    self.axis_index_groups = axis_index_groups
    self.alpha = alpha

    if isinstance(x_example, int):
        feature_shape = [x_example]
    else:
        feature_axes = _canonicalize_axes(x_example.ndim, self.axis)
        feature_shape = [x_example.shape[ax] for ax in feature_axes]
    
    self.mean = nnx.BatchStat(jnp.zeros(feature_shape, jnp.float32))
    self.var = nnx.BatchStat(jnp.ones(feature_shape, jnp.float32))

    if use_scale:
        self.scale = nnx.Param(scale_init(rngs.params(), feature_shape, param_dtype))
    else:
        self.scale = None

    if use_bias:
        self.bias = nnx.Param(bias_init(rngs.params(), feature_shape, param_dtype))
    else:
        self.bias = None

  def __call__(self, x: Array, use_running_average: Optional[bool] = None):
    # Merge use_running_average behavior
    if use_running_average is None:
        use_running_average = self.use_running_average
        if use_running_average is None:
            use_running_average = False


    feature_axes = _canonicalize_axes(x.ndim, self.axis)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)
    # feature_shape = [x.shape[ax] for ax in feature_axes]

    if use_running_average:
      mean, var = self.mean.value, self.var.value
    else:
      mean, var = _compute_stats(
          x, reduction_axes,
          axis_name=self.axis_name,
          axis_index_groups=self.axis_index_groups, alpha=self.alpha)

      # Update running stats
      self.mean.value = self.momentum * self.mean.value + (1 - self.momentum) * mean
      self.var.value = self.momentum * self.var.value + (1 - self.momentum) * var

    # Apply normalization
    # Prepare params
    scale_val = self.scale.value if self.scale is not None else None
    bias_val = self.bias.value if self.bias is not None else None
    
    return _normalize(x, mean, var, reduction_axes, feature_axes, 
                      self.dtype, self.param_dtype, self.epsilon, 
                      scale_val, bias_val)