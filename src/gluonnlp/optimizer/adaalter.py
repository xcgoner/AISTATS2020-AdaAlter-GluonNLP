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

__all__ = ['AdaAlter']


@register
class AdaAlter(Optimizer):
    """AdaAlter optimizer.
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
        super(AdaAlter, self).__init__(**kwargs)
        self.float_stable_eps = eps

    def create_state(self, index, weight):
        return zeros(weight.shape, weight.context, stype=weight.stype)  # history

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, NDArray))
        assert(isinstance(grad, NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        is_sparse = grad.stype == 'row_sparse'
        history = state

        if is_sparse:
            kwargs = {'epsilon': self.float_stable_eps,
                      'rescale_grad': self.rescale_grad}
            if self.clip_gradient:
                kwargs['clip_gradient'] = self.clip_gradient
            sparse.adaalter_update(weight, grad, history, out=weight, lr=lr, wd=wd, **kwargs)
            # raise NotImplementedError('AdaAlter has not been implemented for sparse nd')
        else:
            grad = grad * self.rescale_grad
            if self.clip_gradient is not None:
                grad = clip(grad, -self.clip_gradient, self.clip_gradient)
            div = grad / sqrt(history + self.float_stable_eps)
            weight[:] += (div + weight * wd) * -lr

            history[:] += square(grad)

# Wrapper to inject Horovod broadcast after parameter initialization
def _append_broadcast_init(param, root_rank):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        if self._stype == 'default':
            broadcast_(self.data(), root_rank=root_rank)
            self.data().wait_to_read()
        elif self._stype == 'row_sparse':
            broadcast_(self.row_sparse_data(), root_rank=root_rank)
            self.row_sparse_data().wait_to_read()
        else:
            raise NotImplementedError('DistributedRspTrainer has not been implemented for {} nd'.format(self._stype))
    return wrapped_init_impl


def broadcast_parameters(params, root_rank=0):
    """
    Broadcasts the parameters from root rank to all other processes.
    Typical usage is to broadcast the `Module.get_params()` or the
    `Block.collect_params()`.

    Arguments:
        params: One of the following:
            - dict of parameters to broadcast
            - ParameterDict to broadcast
        root_rank: The rank of the process from which parameters will be
                   broadcasted to all other processes.
    """
    tensors = []
    if isinstance(params, dict):
        tensors = [p for _, p in sorted(params.items())]
    elif isinstance(params, mx.gluon.parameter.ParameterDict):
        for _, p in sorted(params.items()):
            try:
                if p._stype == 'default':
                    tensors.append(p.data())
                elif p._stype == 'row_sparse':
                    tensors.append(p.row_sparse_data())
                else:
                    raise NotImplementedError('DistributedRspTrainer has not been implemented for {} nd'.format(p._stype))
            except mx.gluon.parameter.DeferredInitializationError:
                # Inject wrapper method with post-initialization broadcast to
                # handle parameters with deferred initialization
                new_init = _append_broadcast_init(p, root_rank)
                p._init_impl = types.MethodType(new_init, p)
    else:
        raise ValueError('invalid params of type: %s' % type(params))

    # Run broadcasts.
    for i, tensor in enumerate(tensors):
        broadcast_(tensor, root_rank, str(i))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()