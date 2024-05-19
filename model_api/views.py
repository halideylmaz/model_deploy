# views.py
import json
from django.http import JsonResponse, HttpResponseBadRequest
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os

# Load your models
modelcigarette = YOLO("./best.pt")
modelgun = YOLO("./gun_model.pt")

def process_image(image):
    try:
        result1 = modelcigarette.predict(image, classes=0, conf=0.70, augment=True)
        boxes_cigarette = result1[0].boxes
        result2 = modelgun.predict(image, classes=0, conf=0.60, augment=True)
        boxes_gun = result2[0].boxes

        if boxes_gun and boxes_cigarette:
            return {'gun': True, 'cigarette': True}
        elif boxes_gun:
            return {'gun': True, 'cigarette': False}
        elif boxes_cigarette:
            return {'gun': False, 'cigarette': True}
        else:
            return {'cigarette': False, 'gun': False}
    except Exception as e:
        return {'error': str(e)}

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return HttpResponseBadRequest('Image file not provided')

        image_data = request.FILES['image'].read()
        image = Image.open(BytesIO(image_data))

        result = process_image(image)
        return JsonResponse(result)

    def get(self, request, *args, **kwargs):
        return JsonResponse({'message': 'This is a GET request'})
