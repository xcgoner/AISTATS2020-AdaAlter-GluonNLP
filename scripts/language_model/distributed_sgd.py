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

# coding: utf-8
# pylint: disable=line-too-long
"""Parameter optimizer."""

from mxnet import optimizer as opt
from mxnet.model import _create_kvstore, _create_sparse_kvstore
from mxnet.gluon.parameter import ParameterDict, Parameter

import mxnet as mx
import types
import warnings
import math

import horovod.mxnet as hvd
from horovod.mxnet.mpi_ops import allreduce, allreduce_, allreduce_rsp, broadcast_, broadcast_rsp

class DistributedRspTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None):
        if isinstance(optimizer, DistributedRspTrainer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedRspTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedRspTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params)

    def _allreduce_grads(self):
        # for i, param in enumerate(self._params):
        #     if param.grad_req != 'null':
        #         if param.list_grad()[0].stype == 'default':
        #             allreduce_(param.list_grad()[0], average=True,
        #                     name=str(i), priority=-i)
        #         elif param.list_grad()[0].stype == 'row_sparse':
        #             param.list_grad()[0] = allreduce_rsp(param.list_grad()[0], average=True,
        #                                                  name=str(i), priority=-i)
        #         else:
        #             raise NotImplementedError('DistributedRspTrainer has not been implemented for {} nd'.format(param.list_grad()[0].stype))

# Wrapper to inject Horovod broadcast after parameter initialization
def _append_broadcast_init(param, root_rank):
    init_impl = getattr(param, '_init_impl')
    def wrapped_init_impl(self, *args, **kwargs):
        init_impl(*args, **kwargs)
        if self._stype == 'default':
            broadcast_(self.data(), root_rank=root_rank)
            self.data().wait_to_read()
        elif self._stype == 'row_sparse':
            data = self.row_sparse_data()
            data = broadcast_rsp(self.row_sparse_data(), root_rank=root_rank)
            data.wait_to_read()
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
        if tensor.stype == 'default':
            broadcast_(tensor, root_rank, str(i))
        elif tensor.stype == 'row_sparse':
            tensor = broadcast_rsp(tensor, root_rank=root_rank)
        else:
            raise NotImplementedError('DistributedRspTrainer has not been implemented for {} nd'.format(tensor.stype))

    # Make sure tensors pushed to MXNet engine get processed such that all
    # workers are synced before starting training.
    for tensor in tensors:
        tensor.wait_to_read()
