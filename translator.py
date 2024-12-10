from hailo_sdk_client import ClientRunner
import numpy as np

input_size = 640  # 模型输入的尺寸
chosen_hw_arch = "hailo8l"  # 要使用的 Hailo 硬件架构，这里是 Hailo-8L
onnx_model_name = "yolov8n"  # 模型的名字
onnx_path = "yolov8n.onnx"  # 模型的路径
hailo_model_har_path = f"{onnx_model_name}_hailo_model.har"  # 转换后模型的保存路径
hailo_quantized_har_path = f"{onnx_model_name}_hailo_quantized_model.har"  # 量化后模型的保存路径
hailo_model_hef_path = f"{onnx_model_name}.hef"  # 编译后模型的保存路径
images_path = "data/images"  # 图像路径
dataset_output_path = "calib_set.npy"  # 处理完成后的保存路径

# 将 onnx 模型转为 har
runner = ClientRunner(hw_arch=chosen_hw_arch)
hn, npz = runner.translate_onnx_model(model=onnx_path, net_name=onnx_model_name, start_node_names=["images"], end_node_names=["/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv"], net_input_shapes={"images": [1, 3, input_size, input_size]})
runner.save_har(hailo_model_har_path)

# 量化模型
calib_dataset = np.load('calib_set.npy')
runner = ClientRunner(har=hailo_model_har_path)
alls_lines = [
    'model_optimization_flavor(optimization_level=1, compression_level=2)',
    'resources_param(max_control_utilization=0.6, max_compute_utilization=0.6, max_memory_utilization=0.6)',
    'performance_param(fps=5)'
]
runner.load_model_script('\n'.join(alls_lines))
runner.optimize(calib_dataset)
runner.save_har(hailo_quantized_har_path)

# 编译为 hef
runner = ClientRunner(har=hailo_quantized_har_path)
compiled_hef = runner.compile()
with open(hailo_model_hef_path, "wb") as f:
    f.write(compiled_hef)
