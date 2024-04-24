from django.http import JsonResponse
from django.views import View
from PIL import Image
from torchvision import transforms
import torch
import io

from ultralytics import YOLO

# Load your model
model = YOLO('./cigarette_model1.pt')

# Define your image processing function

# Define your Django view
class PredictView(View):
    def post(self, request, *args, **kwargs):
        # Get the image from the request
        image = request.FILES['image'].read()
        # Process the image
       
        # Use your model for inference
        prediction = model.predict(image, conf=0.40, save_crop=True, classes=0, augment=True)
        boxes = prediction[0].boxes
        if boxes:
            return JsonResponse({'prediction': True})
        else:
            return JsonResponse({'prediction': False})
                                
    
    def get(self, request, *args, **kwargs):
        return JsonResponse({'message': 'This is a GET request'})
#i wanna test if it is working
# Load a sample image
image = Image.open('D:/Users/halide/data/datasets/coco8/images/test/image_524.jpg')
# test the model
results = model.predict(image, conf=0.40, classes=0, augment=True, show=True)

boxes = results[0].boxes
if boxes:
    print('Object detected')
else:
    print('No object detected')

