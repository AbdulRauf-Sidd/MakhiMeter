from django.shortcuts import render
from .helpers import *
import cv2
import numpy as np
from .models import Gender
from PIL import Image
from io import BytesIO
from django.core.files.base import ContentFile

# Create your views here.

# Create your views here.
def gender_upload(request):
    if request.method == 'GET':
        return render(request, 'gender_identification_upload.html')
    
    if request.method == 'POST':
        img = request.FILES['img']
        file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        gender = process_new_image(opencv_image)
        if gender == 1:
            return render(request, 'gender_identification_upload.html', {'error': 'Please upload image of drosophila'})

        gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.resize(gray_image, (128, 128)) 

        # user = request.user

        rf, hog_param = load_random_forest("gender/models/log_rf__combined_200_49_8065_hog_drosophila_model.pkl", "gender/models/svg_params.pkl")
        hog, hog_image = extract_hog_features(gray_image, hog_param)

        lbp, hog_param2 = load_local_binary_pattern("gender/models/LPB_MODEL/best_svm_gender_model.pkl", "gender/models/LPB_MODEL/feature_extraction_params.pkl")
        # hog2, hog_image2 = extract_hog_features(gray_image, hog_param2)
        lbp_pred, lbp_score = lbp_preprocess_and_predict(model=lbp, image=gray_image, params=hog_param2)

        
        if len(hog_image.shape) == 2:  # Grayscale
            image = Image.fromarray(hog_image, mode='L')
        elif len(hog_image.shape) == 3 and hog_image.shape[2] == 3:  # RGB
            image = Image.fromarray(hog_image, mode='RGB')

        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_file = ContentFile(buffer.getvalue(), name=f"{img.name}_hog.png")

        rf_pred, rf_score = rf_preprocess_and_predict(rf, hog)

        # print(lbp_pred, rf_pred)

        # fused_score = fuse_probabilities(lbp_pred=rf_pred, lbp_score=lbp_score, rf_score=rf_score)   
        # if fused_score[0] > 0.5:
        #     gender = "Male"
        # else:
        #     gender= 'Female'

        # highest_score = max(fused_score)

        g = Gender.objects.create(
            # user=user,
            image=img, 
            classification=rf_pred,
            hog=image_file,
        )
        
        return render(request, 'gender_identification_output.html', {"hog_map": g.hog.url, "gender_img": g.image.url, 'prediction': rf_pred, 'score': max(rf_score)})


        
    

def gender_output(request):
    if request.method == 'GET':
        context = {
            'url': '/static/images/flight_output_image.jpg'
        }
        return render(request, 'gender_identification_output.html', context)