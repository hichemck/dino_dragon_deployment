
import tensorflow as tf
import tensorflow.lite as tflite
import numpy as np
from io import BytesIO
from urllib import request
from PIL import Image

interpreter = tflite.Interpreter(model_path = 'dino_dragon_10_0.899.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x/255.0

def predict(url):
    target_size = (150,150)
    img = download_image(url)
    img = prepare_image(img, target_size)    

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    pred = interpreter.get_tensor(output_index)
    return pred

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result