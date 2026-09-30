import onnx
import numpy as np

model = onnx.load("./best.onnx")
print="ONNX 输入信息："
for inp in model.graph.input:
    print(inp.name, inp.type)

print("\nONNX 输出信息：")
for out in model.graph.output:
    print(out.name, out.type)
