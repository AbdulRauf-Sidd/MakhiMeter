from .helpers import *
from django.shortcuts import render
import numpy as np
from .models import Wing
from django.core.files.base import ContentFile
from .models import Wing
from .forms import ImageUploadForm
from io import BytesIO
import base64
from PIL import Image, ImageOps

def home(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    
def login(request):
    if request.method == 'GET':
        return render(request, 'login_user.html')
    
def logout(request):
    if request.method == 'GET':
        return render(request, 'logout.html')
    
def wing_upload(request):
    if request.method == 'GET':
        return render(request, 'wing_segmentation_upload.html') 
    if request.method == 'POST':
        
        image = request.FILES.get('img')
        
        # try:
        #     fly_id = request.POST.get('fly-id')
        # except:
        #     fly_id = None

        pre_processed_image = preprocess_image(image, (256, 256))
        processed_image = process_image(pre_processed_image)
        segmented_image = predict(processed_image)
        labeled_image, areas = post_process(segmented_image, (1250, 1950))
     


        buffer_labeled = BytesIO()
        labeled_image.save(buffer_labeled, format="PNG")
        buffer_labeled.seek(0)
        
        # labeled_image_file = ContentFile(buffer_labeled.read(), name=f"{fly_id}_labeled.png")
        labeled_image_file = ContentFile(buffer_labeled.read(), name=f"labeled.png")


        buffer_original = BytesIO()
        with Image.open(image) as img:
            img_no_exif = ImageOps.exif_transpose(img)  # Neutralize EXIF-based transformations
            img_gray = img_no_exif.convert('L')        # Convert to grayscale
            img_resized = img_gray.resize((720, 480), Image.Resampling.LANCZOS)
            img_resized.save(buffer_original, format="PNG")
        buffer_original.seek(0)
        # original_image_file = ContentFile(buffer_original.read(), name=f"{fly_id}_original.png")
        original_image_file = ContentFile(buffer_original.read(), name=f"original.png")

    
        # Save data to the database
        wing = Wing.objects.create(
            # fly_id=fly_id,
            segmented_image=labeled_image_file,  # Save labeled image
            original_image=original_image_file,  # Save original image
            area_2P =  areas['2P'],
            area_3P = areas['3P'],
            area_M = areas['M'],
            area_S = areas['S'],
            area_D = areas['D'],
            area_1P =  areas['1P'],
            area_B1 =  areas['B1']
        )
        
        wing.save()
        
        area_data = [
            {"segment": "2P", "segment_name": "2nd posterior cell", "area": wing.area_2P},
            {"segment": "3P", "segment_name": "3rd posterior cell", "area": wing.area_3P},
            {"segment": "M", "segment_name": "Marginal cell", "area": wing.area_M},
            {"segment": "S", "segment_name": "Submarginal cell", "area": wing.area_S},
            {"segment": "D", "segment_name": "Discal cell", "area": wing.area_D},
            {"segment": "1P", "segment_name": "1st posterior cell", "area": wing.area_1P},
            {"segment": "B1", "segment_name": "Basal cell 1", "area": wing.area_B1},
        ]

        # Render the template and pass the base64 image string
        return render(request, 'wing_output.html', {
            'image_url': wing.segmented_image.url,
            'area_data': area_data,
        })
    
