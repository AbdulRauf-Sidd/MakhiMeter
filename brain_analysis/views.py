from django.shortcuts import render
import os
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import cv2
import numpy as np
from django.http import JsonResponse
from .helpers import detect_circles, detect_circles_and_calculate_area, remove_background_and_calculate_area, draw_circles_on_image


# Create your views here.
def brain_home(request):
    if request.method == 'GET':
        return render(request, 'brain.html')
    if request.method == 'POST':
        uploaded_file = request.FILES.get('img')
        print('Uploaded file:', uploaded_file)
        if uploaded_file:

            # Read uploaded file into a numpy array
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Ensure the static/images directory exists
            save_dir = os.path.join('wing_segmentation', 'static', 'images')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'orig.png')

            # Resize image to 1024x720
            img_resized = cv2.resize(img, (1440, 720))

            # Save the resized original image
            cv2.imwrite(save_path, img_resized)

            # Convert to grayscale and apply binary threshold
            gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            bin_save_path = os.path.join(save_dir, 'bin.png')
            cv2.imwrite(bin_save_path, binary)

            # Save the image using OpenCV
            cv2.imwrite(save_path, img)

        return render(request, 'brain_threshold.html', {})


def circle_image(request):
    if request.method == 'GET':
        detect_circles(
            'wing_segmentation/static/images/bin.png',
            'wing_segmentation/static/images/circles.png'
        )
        return render(request, 'brain_threshold_2.html', {})
    if request.method == 'POST':
        dp = float(request.POST.get('dp', 1))
        param1 = int(request.POST.get('p1', 128))
        param2 = int(request.POST.get('p2', 128))

        detect_circles(
            'wing_segmentation/static/images/bin.png',
            'wing_segmentation/static/images/circles.png',
            dp=dp,
            param1=param1,
            param2=param2
        )
        return render(request, 'brain_threshold_2.html', {})

def brain_output(request):
    if request.method == 'POST':
        dp = float(request.POST.get('dp', 1))
        param1 = int(request.POST.get('p1', 10))
        param2 = int(request.POST.get('p2', 5))
        circle_area, lis = detect_circles_and_calculate_area(
            'wing_segmentation/static/images/bin.png',
            'wing_segmentation/static/images/brain_output.png',
            dp=dp,
            param1=param1,
            param2=param2
        )
        full_area = remove_background_and_calculate_area(
            'wing_segmentation/static/images/orig.png'
        )
        draw_circles_on_image(
            'wing_segmentation/static/images/orig.png',
            lis,
            'wing_segmentation/static/images/circles_drawn.png'
        )
        ratio = circle_area / full_area if full_area > 0 else 0
        return render(request, 'brain_output.html', {
            'circle_area': float(f"{circle_area:.4g}"),
            'full_area': float(f"{full_area:.4g}"),
            'ratio': float(f"{ratio:.4g}")
        })
    
def threshold_image(request):
    if request.method == 'POST':
        print('1')
        threshold_value = int(request.POST.get('threshold', 128))
        # Load your image here
        print(threshold_value)
        image = cv2.imread('wing_segmentation/static/images/orig.png', cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (1440, 720))
        print('3')
        _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
        print('4')
        # Save the binary image temporarily
        cv2.imwrite('wing_segmentation/static/images/bin.png', binary_image)
        print('5')
        return JsonResponse({'status': 'success'})
    return render(request, 'threshold.html')