import numpy as np
import os
import sys
import tvm
import tvm.contrib.graph_executor as graph_executor
import onnxruntime as ort

def load_run_libs(op, target):
    op_name, input1_shape, input2_shape, _ = op
    # load the module back.
    x = np.random.uniform(0, 10, size=input1_shape).astype(np.float32)
    y = np.random.uniform(0, 10, size=input2_shape).astype(np.float32)
    dylib_path = os.path.join(f"./tar/{op_name}-{input1_shape}-{input2_shape}-{target}.tar")
    loaded_lib = tvm.runtime.load_module(dylib_path)
    if target == "rocm":
        dev = tvm.device("rocm", 0)
    elif target == "cuda":
        dev = tvm.device("cuda", 0)
    else:
        dev = tvm.device("cpu", 0)
    tvm_model = graph_executor.GraphModule(loaded_lib["default"](dev))
    tvm_model.set_input("input1", x)
    tvm_model.set_input("input2", y)
    try:
        tvm_model.run()
    except Exception as e:
        sys.exit(2)
    gpu_output = tvm_model.get_output(0)

    session = ort.InferenceSession(os.path.join(f"./onnx/{op_name}-{input1_shape}-{input2_shape}.onnx"), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"input1": x,"input2":y})
    onnx_output = outputs[0]
    
    # print("推理结果:", output)
    # print("模型的输入x为：")
    # print(x)
    # print("模型的输入y为：")
    # print(y)
    # print("输入为:")
    # print(d)
    # print("")
    # print("标准输出:")
    # print(onnx_output)
    # print("软核输出:")
    # print(gpu_output)
    tvm_np = gpu_output.numpy()
    onnx_np = onnx_output
    
    # 比较逻辑
    print(f"{op_name} test!")
    try:
        assert tvm_np.shape == onnx_np.shape
        is_close = np.allclose(tvm_np, onnx_np, atol=1e-4, rtol=1e-4)
        if is_close:
            print("✅ test pass")
            # sys.exit(0)  # 明确成功退出码
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
    
    ops = [
    #     ("05add", (2, 2), (2, 2), "add"),
    ("11matmul", (2, 2), (2, 2), "matmul"),
        ("11matmul", (2, 8), (8, 10), "matmul"),

          ]
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    print("----------------build end--------------\n")
    for op in ops:
        load_run_libs(op, "rocm")