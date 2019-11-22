#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# -*- coding: utf-8 -*-
import json
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.loss import YoloLoss
import os
import yolo
from keras.models import load_model


# %%


from yolo.frontend import create_exist_yolo
from yolo.frontend import create_yolo

# 1. create yolo instance
yolo_detector = create_yolo("Full Yolo", ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], 416)
# yolo_loss = YoloLoss(nb_class=10)
# custom_loss = yolo_loss.custom_loss(16)
# model = load_model('model.h5', custom_objects={'loss_func': custom_loss})
# yolo_detector = create_exist_yolo("Full Yolo", model, ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], 416)
# yolo_detector.summary()


# %%


# 2. load pretrained weighted file
# Pretrained weight file is at https://drive.google.com/drive/folders/1Lg3eAPC39G9GwVTCH3XzF73Eok-N-dER
import json
DEFAULT_WEIGHT_FILE = os.path.join('svhn', "weights.h5")
print(DEFAULT_WEIGHT_FILE)
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)


# %%


# 3. Load images

# import os
# import matplotlib.pyplot as plt
# %matplotlib inline  
# DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "tests", "dataset", "svhn", "imgs")

# img_files = [os.path.join(DEFAULT_IMAGE_FOLDER, "1.png"), os.path.join(DEFAULT_IMAGE_FOLDER, "2.png")]
# imgs = []
# for fname in img_files:
#     img = cv2.imread(fname)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
# #     img = cv2.resize(img, (416, 416), interpolation=cv2.INTER_CUBIC)

#     imgs.append(img)
#     plt.imshow(img)
#     plt.show()
    
# print(imgs[0].type)


# %%


# 4. Predict digit region
# import numpy as np

# THRESHOLD = 0.3
# json_dic = []

# for img in imgs:
#     boxes, probs, confidences  = yolo_detector.predict(img, THRESHOLD)

# #     4. save detection result
#     image, dic = draw_scaled_boxes(img,
#                               boxes,
#                               probs,
#                               ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
#                              confidences)
#     print(dic)
#     json_dic.append(dic)
#     print("{}-boxes are detected.".format(len(boxes)))
#     plt.imshow(image)
#     plt.show()
    
# data = json.dumps(json_dic)
# with open('data.json', 'w') as fp:
#     json.dump(json_dic, fp)


# %%


# 4. Predict digit region
import numpy as np
import os
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
images_dir = os.path.join('./test')
THRESHOLD = 0.3
lst = os.listdir(images_dir)
lst.sort()
json_dic = []
for i in range(13068):
    
    file_name = str(i+1) + ".png"
    print(file_name)
    image_path = os.path.join(images_dir, file_name)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    

    boxes, probs, confidences  = yolo_detector.predict(img, THRESHOLD)

    #     4. save detection result
    image, dic = draw_scaled_boxes(img,
                              boxes,
                              probs,
                              ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                             confidences)

    print("{}-boxes are detected.".format(len(boxes)))
    print(dic)
    json_dic.append(dic)
#     plt.imshow(image)
#     plt.show()
    
# data = json.dumps(json_dic)
with open('data.json', 'w') as fp:
    json.dump(json_dic, fp)


# %%




