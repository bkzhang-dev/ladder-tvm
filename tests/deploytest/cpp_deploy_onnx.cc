/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdio>
using namespace std;
void DeployGraphExecutor() {
  LOG(INFO) << "Running graph executor...";
  // load in the library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("lib/test_onnx_addrelu.so");
  // create the graph executor module
  tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = gmod.GetFunction("run");

  // Use the C++ API
  tvm::runtime::NDArray x = tvm::runtime::NDArray::Empty({1, 1, 3, 3}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray y = tvm::runtime::NDArray::Empty({1, 1, 3, 3}, DLDataType{kDLFloat, 32, 1}, dev);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      static_cast<float*>(x->data)[i * 3 + j] = i * 3 + j;
    }
  }
  static_cast<float*>(x->data)[5] = -5;
  // set the right input
  set_input("input", x);
  // run the code
  run();
  // get the output
  get_output(0, y);

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      cout<<static_cast<float*>(x->data)[i * 3 + j]<<" ";      
    }
  }
  cout << endl;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
    cout<<static_cast<float*>(y->data)[i * 3 + j]<<" ";    
   }
  }

}

int main(void) {
  DeployGraphExecutor();
  return 0;
}
