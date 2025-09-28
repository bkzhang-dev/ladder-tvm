import numpy as np
import tvm
import tvm.contrib.graph_executor as graph_executor
import os
import onnxruntime as ort
import sys
import ast

def load_run_libs(input1_shape, input2_shape):
    # load the module back.
    x = np.random.randn(*input1_shape).astype(np.float32)
    # y = np.random.randn(*input2_shape).astype(np.float32)

    dylib_path = f"./01matmul.tar"
    loaded_lib = tvm.runtime.load_module(dylib_path)
    dev = tvm.device("rocm", 0)
    tvm_model = graph_executor.GraphModule(loaded_lib["default"](dev))
    tvm_model.set_input("input1", x)
    # tvm_model.set_input("input2", y)
    tvm_model.run()
    gpu_output = tvm_model.get_output(0)

    session = ort.InferenceSession(f"../onnx/19gemm.onnx", providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input1":x})
    onnx_output = outputs[0]
    
    print("模型的输入x为：")
    print(x)
    print("模型的输入y为：")
    print(y)
    # print("")
    print("标准输出:")
    print(onnx_output)
    print("软核输出:")
    print(gpu_output)
    tvm_np = gpu_output.numpy()
    print(tvm_np)
    onnx_np = onnx_output
    
    # 比较逻辑
    print("11matmul test!")
    try:
        assert tvm_np.shape == onnx_np.shape
        is_close = np.allclose(tvm_np, onnx_np, atol=1e-5, rtol=1e-5)
        if is_close:
            print("✅ test pass")
            sys.exit(0)  # 明确成功退出码
        else:
            print("❌ output error")
            print("expected output:")
            print(onnx_np)
            print("but get:")
            print(tvm_np)
            sys.exit(1)  # 失败退出码
    except AssertionError as e:
        print(f"output shape error: {e}")
        sys.exit(1)  # 失败退出码


if __name__ == "__main__":
    input1_shape, input2_shape = ((16, 16), (16, 16))
    load_run_libs(input1_shape, input2_shape)
