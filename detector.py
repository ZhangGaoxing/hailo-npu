import cv2
import numpy as np
from hailo_platform import HEF, Device, VDevice, InputVStreamParams, OutputVStreamParams, FormatType, HailoStreamInterface, InferVStreams, ConfigureParams

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
               'hair drier', 'toothbrush']

colors = np.random.uniform(0, 255, size=(len(class_names), 3))

# 根据坐标画出检测框
def draw_bboxes(image, bboxes, confidences, class_ids, class_names, colors):
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        label = f'{class_names[class_ids[i]]}: {confidences[i]:.2f}'
        color = colors[class_ids[i]]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 加载YOLOv8模型
hef_path = 'yolov8s.hef'
hef = HEF(hef_path)
# 初始化Hailo设备
devices = Device.scan()
target = VDevice(device_ids=devices)
# 配置网络组
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = target.configure(hef, configure_params)[0]
network_group_params = network_group.create_params()
# 获取输入输出流信息
input_vstream_info = hef.get_input_vstream_infos()[0]
output_vstream_info = hef.get_output_vstream_infos()[0]
# 创建输入输出虚拟流参数
input_vstreams_params = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
output_vstreams_params = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

# 使用摄像头0作为视频源
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 对图像进行预处理
    resized_frame = cv2.resize(frame, (input_vstream_info.shape[0], input_vstream_info.shape[1]))
    input_data = {input_vstream_info.name: np.expand_dims(np.asarray(resized_frame), axis=0).astype(np.float32)}
    # 创建输入输出虚拟流并推理
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params, tf_nms_format = True) as infer_pipeline:
        with network_group.activate(network_group_params):
            output_data = infer_pipeline.infer(input_data)

    # 图像缩放比例
    scale_x = frame.shape[1] / input_vstream_info.shape[1]
    scale_y = frame.shape[0] / input_vstream_info.shape[0]

    # 提取边界框、类别等信息，并在原始帧上绘制
    for key in output_data.keys():
        num_classes, bbox_params, num_detections = output_data[key][0].shape

        boxes = []
        confidences = []
        class_ids = []

        for class_id in range(num_classes):
            for detection_id in range(num_detections):
                bbox = output_data[key][0][class_id, :, detection_id]
                if bbox[4] > 0.5:
                    x1, y1, x2, y2, confidence = bbox[:5]

                    x1 = int(x1 * input_vstream_info.shape[0] * scale_x)
                    y1 = int(y1 * input_vstream_info.shape[1] * scale_y)
                    x2 = int(x2 * input_vstream_info.shape[0] * scale_x)
                    y2 = int(y2 * input_vstream_info.shape[1] * scale_y)

                    print(f'{class_names[class_id]}: {[x1, y1, x2, y2]} {bbox[:5]}')

                    boxes.append([x1, y1, x2, y2])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        draw_bboxes(frame, boxes, confidences, class_ids, class_names, colors)

    cv2.imshow('Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
