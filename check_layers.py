import onnx

onnx_model_path = "yolov8n.onnx"
model = onnx.load(onnx_model_path)

print("Input Nodes:")
for input in model.graph.input:
    print(input.name)

print("\nOutput Nodes:")
for output in model.graph.output:
    print(output.name)

print("\nNodes:")
for node in model.graph.node:
    print(node.name)