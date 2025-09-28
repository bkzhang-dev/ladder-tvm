import sys
import os
import numpy as np
import tvm
import tvm.contrib.graph_executor as graph_executor
from PIL import Image

def preprocess_image(image_path):
    """返回形状为 [784] 的1D数组"""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))
    img_np = np.array(img, dtype=np.float32) / 255.0  # 避免DeprecationWarning
    img_np = (img_np - 0.5) / 0.5
    return img_np.flatten()  # 确保是 [784]

def run_mnist_model(image_path, target="cpu"):  # 默认用CPU
    input_data = preprocess_image(image_path)    # [784]
    
    # 加载模型
    dylib_path = "./mlp.tar"
    loaded_lib = tvm.runtime.load_module(dylib_path)
    dev = tvm.device(target, 0)
    tvm_model = graph_executor.GraphModule(loaded_lib["default"](dev))
    
    # 设置输入并运行
    tvm_model.set_input("input", input_data)  # 直接传入1D数组
    tvm_model.run()
    output = tvm_model.get_output(0).numpy()
    
    print("\n=== 预测结果 ===")
    print("输入图片:", image_path)
    print("预测类别:", np.argmax(output))
    print("各类别概率:", {i: f"{p:.4f}" for i, p in enumerate(output)})

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python test_mlp.py <图片路径>")
        sys.exit(1)
    
    test_image = sys.argv[1]
    if not os.path.exists(test_image):
        print(f"错误：图片文件 {image_path} 不存在！", file=sys.stderr)
        sys.exit(1)
    run_mnist_model(test_image, target="rocm")  # 改用CPU避免ROCm问题
