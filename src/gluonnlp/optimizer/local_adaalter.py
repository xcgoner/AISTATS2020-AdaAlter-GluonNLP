# coding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""AdaAlter optimizer"""

from mxnet.optimizer import Optimizer, register
from mxnet.ndarray import zeros, NDArray
from mxnet.ndarray import square, power, sqrt, maximum, minimum, clip
from mxnet.ndarray import sparse

__all__ = ['LocalAdaAlter']


@register
class LocalAdaAlter(Optimizer):
    """LocalAdaAlter optimizer.
    TODO(xcong): update the description
    This class implements the AdaGrad optimizer described in *Adaptive Subgradient
    Methods for Online Learning and Stochastic Optimization*, and available at
    http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf.
    This optimizer updates each weight by::
        grad = clip(grad * rescale_grad, clip_gradient)
        div = grad / sqrt(history + float_stable_eps)
        weight += (div + weight * wd) * -lr
        history += square(grad)
    This optimizer accepts the following parameters in addition to those accepted
    by :class:`.Optimizer`.
    See Also
    ----------
    :meth:`mxnet.ndarray.sparse.adagrad_update`.
    Parameters
    ----------
    eps: float, optional
        Initial value of the history accumulator. Avoids division by 0.
    """
    def __init__(self, eps=1e-7, **kwargs):
        super(LocalAdaAlter, self).__init__(**kwargs)
        self.float_stable_eps = eps
        self._local_sgd_counter = 0

    def create_state(self, index, weight):
        return (zeros(weight.shape, weight.context, stype=weight.stype),   # history
                zeros(weight.shape, weight.context, stype=weight.stype))

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        is_sparse = grad.stype == 'row_sparse'
        history = state[0]
        cache_history = state[0]

        if is_sparse:
            kwargs = {'epsilon': self.float_stable_eps * self._local_sgd_counter,
                      'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            sparse.adaalter_update(weight, grad, cache_history, out=weight, lr=lr, wd=wd, **kwargs)
            # raise NotImplementedError('AdaAlter has not been implemented for sparse nd')
        else:
            grad[:] = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad[:] = clip(grad, -self.clip_gradient, self.clip_gradient)
            div = grad / sqrt(history + self.float_stable_eps * self._local_sgd_counter)
            weight[:] += (div + weight * wd) * -lr

            cache_history[:] += square(grad)
