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
"""Script to prepare test_addone.so"""
import tvm
import onnx
import numpy as np
from tvm import te
from tvm import relay
import os


def prepare_test_libs(base_path):
    onnx_model = onnx.load("test_model.onnx")


    data = np.empty([1,1,3,3], dtype = float) 
    target = "llvm"

    input_name = "input"
    shape_dict = {input_name: data.shape}

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    compiled_lib = relay.build(mod, tvm.target.Target("llvm"), params=params)
    dylib_path = os.path.join(base_path, "test_onnx_addrelu.so")
    compiled_lib.export_library(dylib_path)


# def prepare_graph_lib(base_path):
#     x = relay.var("x", shape=(2, 2), dtype="float32")
#     y = relay.var("y", shape=(2, 2), dtype="float32")
#     params = {"y": np.ones((2, 2), dtype="float32")}
#     mod = tvm.IRModule.from_expr(relay.Function([x, y], x + y))
#     # build a module
#     compiled_lib = relay.build(mod, tvm.target.Target("llvm"), params=params)
#     # export it as a shared library
#     # If you are running cross compilation, you can also consider export
#     # to tar and invoke host compiler later.
#     dylib_path = os.path.join(base_path, "test_relay_add.so")
#     compiled_lib.export_library(dylib_path)


if __name__ == "__main__":
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    prepare_test_libs(os.path.join(curr_path, "lib"))
    # prepare_graph_lib(os.path.join(curr_path, "lib"))
