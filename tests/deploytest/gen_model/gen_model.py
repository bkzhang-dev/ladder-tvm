import torch.nn as nn
import torch
import torch.nn.functional as F


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    def forward(self, x):
        # 将两个输入张量相加
        x1 = F.relu(x)
        x = x + x1
        return x


# 创建模型实例

model = TestModel()

out_path = '../onnx_models/relu_add.onnx'
x = torch.randn(1,1,100,100)
torch.onnx.export(model, x,out_path , verbose=True, input_names=['input'], output_names=['output'])
