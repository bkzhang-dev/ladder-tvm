import numpy as np
import tvm
import numpy as np
import tvm.contrib.graph_executor as graph_executor
import os
import onnxruntime as ort
import sys

def load_run_libs(base_path):
    # load the module back.
    input_data = np.random.randn(2,2,28,28).astype("float32")   
    dylib_path = os.path.join(base_path, "cnn.tar")
    loaded_lib = tvm.runtime.load_module(dylib_path)
    dev = tvm.device("rocm", 0)
    # dev = tvm.device("cuda", 0)
    # dev = tvm.device("cpu", 0)
    tvm_model = graph_executor.GraphModule(loaded_lib["default"](dev))
    tvm_model.set_input("input", input_data)
    tvm_model.run()

    session = ort.InferenceSession(os.path.join(base_path, "cnn.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input": input_data})
    onnx_output = outputs[0]
    gpu_output = tvm_model.get_output(0)
    
    # print("推理结果:", output)
    # print("模型的输入x为：")
    # print(x)
    # print("模型的输入y为：")
    # print(y)
    # print("输入为:")
    # print(d)
    # print("标准输出:")
    # print(onnx_output)
    # print("软核输出:")
    # print(gpu_output)
    tvm_np = gpu_output.numpy()
    onnx_np = onnx_output
    print("cnn test!")
    # 比较逻辑
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
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    print("----------------build end--------------\n")
    load_run_libs(curr_path)