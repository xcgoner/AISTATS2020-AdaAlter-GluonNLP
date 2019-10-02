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
from horovod.mxnet.mpi_ops import allreduce, allreduce_

class DistributedRspTrainer(mx.gluon.Trainer):
    def __init__(self, params, optimizer, optimizer_params=None, sdtype='float32'):
        if isinstance(optimizer, DistributedRspTrainer):
            optimizer = optimizer._optimizer
            warnings.warn("DistributedRspTrainer does not take DistributedOptimizer "
                          "as its optimizer. We have unwrapped it for you.")

        super(DistributedRspTrainer, self).__init__(
            params, optimizer, optimizer_params=optimizer_params, kvstore=None, update_on_kvstore = False)
        
        self._update_on_kvstore = False

        self._hvd_param_buf = {}
        self._sdtype = sdtype
        if sdtype == 'float32':
            self._dtype_mismatch = False
        else:
            self._dtype_mismatch = True

    def _allreduce_grads(self):
        # super(DistributedRspTrainer, self)._allreduce_grads()
        # print(self._kvstore)
        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype == 'default':
                    allreduce_(param.list_grad()[0], average=True,
                               name=str(i), priority=-i)
                else:
                    if i not in self._hvd_param_buf:
                        self._hvd_param_buf[i] = mx.nd.zeros(param.list_grad()[0].shape, param.list_grad()[0].context, dtype=self._sdtype)
                    param_dense = self._hvd_param_buf[i]
                    if self._dtype_mismatch:
                        param_dense[:] = param.list_grad()[0].tostype('default')
                    else:
                        mx.nd.sparse.cast_storage(param.list_grad()[0], 'default', out=param_dense)
                    allreduce_(param_dense, average=True,
                                name=str(i), priority=-i)
                    # mx.nd.sparse.cast_storage(param_dense, 'row_sparse', out=param.list_grad()[0])

        for i, param in enumerate(self._params):
            if param.grad_req != 'null':
                if param.list_grad()[0].stype != 'default':
                    if i in self._hvd_param_buf:
                        if self._dtype_mismatch:
                            param.list_grad()[0][:] = self._hvd_param_buf[i].tostype('row_sparse')
                        else:
                            mx.nd.sparse.cast_storage(self._hvd_param_buf[i], 'row_sparse', out=param.list_grad()[0])
