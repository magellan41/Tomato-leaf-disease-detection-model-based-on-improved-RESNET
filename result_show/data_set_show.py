import cv2
import matplotlib.pyplot as plt

# 读取data目录下的文件夹
import os

diseases = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
            'Target_Spot', 'Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_mosaic_virus', 'healthy']

path = '../data/PlantVillage'
files = os.listdir(path)
total = len(files)
plt.subplots(2, int(total / 2), figsize=(20, 8))
plt.tight_layout()
for i in range(total):
    file = files[i]
    # 读取文件夹下的第一个文件
    img_path = os.path.join(path, file)
    imgs = os.listdir(img_path)
    img = os.path.join(img_path, imgs[0])
    img = cv2.imread(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(2, int(total / 2), i + 1)
    plt.imshow(img)
    # 隐藏坐标轴
    plt.axis('off')
    plt.title(diseases[i])
plt.show()
