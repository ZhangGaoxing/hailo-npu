import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

input_size = 640
images_path = "data/images"
dataset_output_path = "calib_set.npy"

images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] in [".jpg", ".png", "bmp"]][:1500]  # 获取图像名称列表
calib_dataset = np.zeros((len(images_list), input_size, input_size, 3))  # 初始化 numpy 数组

for idx, img_name in enumerate(sorted(images_list)):
    img = cv2.imread(os.path.join(images_path, img_name))
    resized = cv2.resize(img, (input_size, input_size))  # 调整原始图像的尺寸为模型输入的尺寸
    calib_dataset[idx,:,:,:]=np.array(resized)
np.save(dataset_output_path, calib_dataset)

def show_images(images, titles=None, cols=5):
    n_images = len(images)
    if titles is None:
        titles = [''] * n_images
    fig, axes = plt.subplots(nrows=(n_images // cols), ncols=cols, figsize=(15, 5))
    for ax, img, title in zip(axes.flat, images, titles):
        ax.imshow(img / 255)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

original_images = [np.array(cv2.imread(os.path.join(images_path, img_name))) for img_name in images_list[:5]]
show_images(original_images, titles=[f"Original {i}" for i in range(1, 6)])

processed_images = [calib_dataset[i] for i in range(5)]
show_images(processed_images, titles=[f"Processed {i}" for i in range(1, 6)])
