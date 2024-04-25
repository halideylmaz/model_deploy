from django.http import JsonResponse, HttpResponseBadRequest
from django.views import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator

from ultralytics import YOLO
from PIL import Image
from io import BytesIO
# Load your model
model = YOLO('./cigarette_model1.pt')


@method_decorator(csrf_exempt, name='dispatch')
class PredictView(View):
    def post(self, request, *args, **kwargs):
        # Check if the 'image' key exists in request.FILES
        if 'image' not in request.FILES:
            return HttpResponseBadRequest('Image file not provided')

        # Get the image from the request
        image_data = request.FILES['image'].read()

        # Convert the image data to a PIL Image object
        image = Image.open(BytesIO(image_data))

        # Use your model for inference
        prediction = model.predict(image, conf=0.10, classes=0, augment=True)
        boxes = prediction[0].boxes
        if boxes:
            return JsonResponse({'prediction': True})
        else:
            return JsonResponse({'prediction': False})

    def get(self, request, *args, **kwargs):
        return JsonResponse({'message': 'This is a GET request'})