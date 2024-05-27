# views.py
import asyncio
import json
from django.http import FileResponse, HttpResponse, HttpResponseRedirect, JsonResponse, HttpResponseBadRequest
from django.shortcuts import render
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import os
from django.http import FileResponse, HttpResponse
from wsgiref.util import FileWrapper  # Import FileWrapper
from django.shortcuts import redirect


# Load your models
modelcigarette = YOLO("./best.pt")
# modelgun = YOLO("./gun_model1.pt")

def index_page(request):
    return render(request, 'model_api/index.html')
"""
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
"""


async def process_image(image):
    try:
        loop = asyncio.get_event_loop()
        result1 = await loop.run_in_executor(None, lambda: modelcigarette.predict(image, classes=0, conf=0.70, augment=True))
        boxes_cigarette = result1[0].boxes
        if boxes_cigarette:
            return {'gun': False, 'cigarette': True}
        return {'gun': False, 'cigarette': False}
    except Exception as e:
        return {'error': str(e)}

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    async def post(self, request, *args, **kwargs):
        if 'image' not in request.FILES:
            return HttpResponseBadRequest('Image file not provided')

        try:
            image_data = request.FILES['image'].read()
            image = Image.open(BytesIO(image_data))
            result = await process_image(image)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    async def get(self, request, *args, **kwargs):
        return JsonResponse({'message': 'This is a GET request'})
    
    
def open_app(request):
    # Render a template containing JavaScript for the redirection
    return render(request, 'model_api/open_app.html')



def download_apk(request):
    google_drive_link = "https://drive.google.com/file/d/1lP9SaGsdcsKmlTggKfNly0nw1Cb4MGJu/view?usp=drive_link"
    return redirect(google_drive_link)
