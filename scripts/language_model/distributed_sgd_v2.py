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
            params, optimizer, optimizer_params=optimizer_params, kvstore='local')

        self._hvd_param_buf = {}

    def _allreduce_grads(self):
        super(DistributedRspTrainer, self)._allreduce_grads()
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype == 'default':
                    allreduce_(param.list_grad()[0], average=True,
                            name=str(i), priority=-i)
                elif param.list_grad()[0].stype == 'row_sparse':
                    param.list_grad()[0] = allreduce_rsp(param.list_grad()[0], average=True,
                                                         name=str(i), priority=-i)
                else:
                    raise NotImplementedError('DistributedRspTrainer has not been implemented for {} nd'.format(param.list_grad()[0].stype))
