from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import tensorflow as tf

from edward.inferences.map import map
from edward.util import get_variables

try:
  from edward.models import \
      MultivariateNormalDiag, MultivariateNormalTriL, Normal
except Exception as e:
  raise ImportError("{0}. Your TensorFlow version is not supported.".format(e))


def laplace(model, variational, align_latent, align_data,
            scale=None, auto_transform=True, collections=None, *args, **kwargs):
  """Laplace approximation [@laplace1986memoir].

  It approximates the posterior distribution using a multivariate
  normal distribution centered at the mode of the posterior.

  We implement this by running `MAP` to find the posterior mode.
  This forms the mean of the normal approximation. We then compute the
  inverse Hessian at the mode of the posterior. This forms the
  covariance of the normal approximation.

  #### Notes

  If `MultivariateNormalDiag` or `Normal` random variables are
  specified as approximations, then the Laplace approximation will
  only produce the diagonal. This does not capture correlation among
  the variables but it does not require a potentially expensive
  matrix inversion.

  Random variables with both scalar batch and event shape are not
  supported as `tf.hessians` is currently not applicable to scalars.

  Note that `Laplace` finds the location parameter of the normal
  approximation using `MAP`, which is performed on the latent
  variable's original (constrained) support. The scale parameter
  is calculated by evaluating the Hessian of $-\log p(x, z)$ in the
  constrained space and under the mode. This implies the Laplace
  approximation always has real support even if the target
  distribution has constrained support.

  #### Examples

  ```python
  X = tf.placeholder(tf.float32, [N, D])
  w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
  y = Normal(loc=ed.dot(X, w), scale=tf.ones(N))

  qw = MultivariateNormalTriL(
      loc=tf.Variable(tf.random_normal([D])),
      scale_tril=tf.Variable(tf.random_normal([D, D])))

  inference = ed.Laplace({w: qw}, data={X: X_train, y: y_train})
  ```
  """
  """Create an inference algorithm.

  Args:
    latent_vars: list of RandomVariable or
                 dict of RandomVariable to RandomVariable.
      Collection of random variables to perform inference on. If list,
      each random variable will be implictly optimized using a
      `MultivariateNormalTriL` random variable that is defined
      internally with unconstrained support and is initialized using
      standard normal draws. If dictionary, each random
      variable must be a `MultivariateNormalDiag`,
      `MultivariateNormalTriL`, or `Normal` random variable.
  """
  # TODO build program runnin variational but replacing with point
  # masses (or just call mean() for value)
  # Store latent variables in a temporary object; MAP will
  # optimize `PointMass` random variables, which subsequently
  # optimizes location parameters of the normal approximations.
  latent_vars_normal = latent_vars.copy()
  latent_vars = {z: PointMass(params=qz.loc)
                 for z, qz in six.iteritems(latent_vars_normal)}

  loss = map(model, variational, align_latent, align_data,
             scale, auto_transform, collections, *args, **kwargs)
  # TODO
  def _finalize(loss, latent_vars, latent_vars_normal):
    """Function to call after convergence.

    Computes the Hessian at the mode.
    """
    hessians = tf.hessians(loss, list(six.itervalues(latent_vars)))
    finalize_ops = []
    for z, hessian in zip(six.iterkeys(latent_vars), hessians):
      qz = latent_vars_normal[z]
      if isinstance(qz, (MultivariateNormalDiag, Normal)):
        scale_var = get_variables(qz.variance())[0]
        scale = 1.0 / tf.diag_part(hessian)
      else:  # qz is MultivariateNormalTriL
        scale_var = get_variables(qz.covariance())[0]
        scale = tf.matrix_inverse(tf.cholesky(hessian))

      finalize_ops.append(scale_var.assign(scale))
    return tf.group(*finalize_ops)
  finalize_op = _finalize(loss, latent_vars, latent_vars_normal)
  return loss, finalize_op
