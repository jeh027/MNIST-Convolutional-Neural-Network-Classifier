from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader
import json
import base64
from PIL import Image
import io
import numpy as np
import matplotlib.pyplot as plt
from .ml_model import load_model

# Create your views here.

loaded_model = load_model()
# print(load_model.summary())

# Preprocessing function
def preprocess_image(image_data):
    
    image_data = base64.b64decode(image_data.split(',')[1])
    
    with open('output_image.png', 'wb') as f:
        f.write(image_data)
    
    image = Image.open("C:/Users/HP/Desktop/MNIST/mnist_project/output_image.png").convert('L')
    image = image.resize((28, 28))  # Resize to 28x28
    img_array = np.array(image) / 255.0  # Normalize pixel values
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the neural network
    # print(img_array)
    return img_array


def predict_digit(request):
    # POST request
    if request.method == 'POST':
        
        data = json.loads(request.body)
        image_data = data.get('image')

        preprocessed_image = preprocess_image(image_data)
        
        prediction = loaded_model.predict(preprocessed_image)
        print(prediction)
        predicted_digit = np.argmax(prediction)
        
        return JsonResponse({'digit': int(predicted_digit)})
    
    
    # GET request loads the canvas
    template = loader.get_template("mnist_app/canvas.html")
    context = {}
    return HttpResponse(template.render(context, request))