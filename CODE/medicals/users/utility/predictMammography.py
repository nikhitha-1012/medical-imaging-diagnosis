import os
import numpy as np
from PIL import Image

from django.conf import settings
from keras.models import load_model
from keras.utils import load_img

# -------------------------
# Load model ONLY ONCE
# -------------------------
MODEL_PATH = os.path.join(settings.MEDIA_ROOT, "mammography_model3.h5")
model = load_model(MODEL_PATH)


def start_process(imagepath):

    img_path = os.path.join(settings.MEDIA_ROOT, imagepath)

    def getCropImgs(img):
        z = np.asarray(img, dtype=np.float32)
        crops = []

        for i in range(3):
            for j in range(4):
                crop = z[
                    512 * i:512 * (i + 1),
                    512 * j:512 * (j + 1),
                    :
                ]
                crops.append(crop)

        return crops

    def softmaxToProbs(soft):
        e = np.exp(soft[0])
        return (e / np.sum(e)) * 100

    def predict(img):

        x = np.expand_dims(img, axis=0)

        pred = model.predict(x, verbose=0)

        probs = softmaxToProbs(pred)

        return probs

    img = load_img(img_path)

    crops = np.array(getCropImgs(img), dtype=np.float32)

    crops = crops / 255.0

    classes = [
        "Benign",
        "InSitu",
        "Invasive",
        "Normal"
    ]

    compProbs = np.zeros(4)

    for crop in crops:

        probs = predict(crop)

        compProbs += probs

    prediction = classes[np.argmax(compProbs)]

    return prediction