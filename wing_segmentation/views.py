from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.uploadedfile import InMemoryUploadedFile
from io import BytesIO
from PIL import Image
import base64
from .helpers import *
from .forms import *

from django.shortcuts import render
from .forms import ImageUploadForm
from PIL import Image, ImageFilter
import numpy as np

from django.http import JsonResponse, HttpResponse
from PIL import Image, ImageFilter
import io

# Import necessary libraries
from django.http import HttpResponse
from PIL import Image, ImageFilter
import io, os

from .models import Wing

def show_image_processor(request):
    return render(request, 'test.html')


def save_image_from_static(request):
    # Set the path to the image
    image_path = 'wing_segmentation/static/images/image0185.png'
    filename = os.path.basename(image_path)

    # Get erosion and dilation values from the request
    erosion_value = int(request.GET.get('erosion', 0))
    dilation_value = int(request.GET.get('dilation', 0))
    tresh_value = int(request.GET.get('treshold', 127))

    if tresh_value == 0:
        tresh_value = 127

    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Step 2: Convert to binary image using thresholding
    _, binary_image = cv2.threshold(gray_image, tresh_value, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary_image)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # Dilation
    dilated_image = cv2.dilate(inverted_image, kernel, iterations=dilation_value)
    eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_value)
    # cv2.imwrite('static/images/perfect binary/'+filename, eroded_image)
    img1 = Image.fromarray(eroded_image).save('wing_segmentation/static/images/perfect binary/'+filename)
    print("success")

    img = cv2.resize(eroded_image, (1080, 720), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(img)

    # Save the processed image to a BytesIO object
    img_io = io.BytesIO()
    img.save(img_io, 'PNG')
    img_io.seek(0)
    # Return the image as a response
    return HttpResponse(img_io.getvalue(), content_type='image/png')


# def process_image_from_static(request):
#     # Set the path to the image
#     image_path = 'wing_segmentation/static/images/image0188.png'

#     # Get erosion and dilation values from the request
#     erosion_value = int(request.GET.get('erosion', 0))
#     dilation_value = int(request.GET.get('dilation', 0))
#     tresh_value = int(request.GET.get('treshold', 127))
#     filename = os.path.basename(image_path)

#     if tresh_value == 0:
#         tresh_value = 127

#     print(erosion_value, dilation_value, tresh_value) 

#     # Open the image
#         # Apply erosion if specified
#     gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     gray_image = cv2.equalizeHist(gray_image)
#     # Step 2: Convert to binary image using thresholding
#     _, binary_image = cv2.threshold(gray_image, tresh_value, 255, cv2.THRESH_BINARY)
#     inverted_image = cv2.bitwise_not(binary_image)
#     # Step 3: Apply dilation and erosion
#     # Define the kernel size
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     # Dilation
#     dilated_image = cv2.dilate(inverted_image, kernel, iterations=dilation_value)
#     eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_value)
#     # if erosion_value > 0:
#     #     img = img.filter(ImageFilter.MinFilter(size=3))  # Simulating erosion
#     # # Apply dilation if specified
#     # if dilation_value > 0:
#     #     img = img.filter(ImageFilter.MaxFilter(size=3))  # Simulating dilation
#     # binary_for_skeleton = cv2.threshold(eroded_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     # skeleton = np.zeros_like(binary_for_skeleton)
#     # elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     # while True:
#     #     open = cv2.morphologyEx(binary_for_skeleton, cv2.MORPH_OPEN, elem)
#     #     temp = cv2.subtract(binary_for_skeleton, open)
#     #     eroded = cv2.erode(binary_for_skeleton, elem)
#     #     skeleton = cv2.bitwise_or(skeleton, temp)
#     #     binary_for_skeleton = eroded.copy()
#     #     if cv2.countNonZero(binary_for_skeleton) == 0:
#     #         break

#     # dilated_skeleton = cv2.dilate(skeleton, kernel, iterations=1)

#     # dist_transform = cv2.distanceTransform(cv2.bitwise_not(skeleton), cv2.DIST_L2, 5)
#     # ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
#     # sure_fg = np.uint8(sure_fg)
#     # unknown = cv2.subtract(cv2.bitwise_not(sure_fg), skeleton)

#     # # Marker labelling
#     # ret, markers = cv2.connectedComponents(sure_fg)

#     # # Add one to all labels so that sure background is not 0, but 1
#     # markers = markers + 1

#     # # Now, mark the region of unknown with zero
#     # markers[unknown == 255] = 0

#     # # Watershed
#     # markers = cv2.watershed(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), markers)
#     # watershed_result = cv2.applyColorMap(np.uint8(255*(markers > 1)), cv2.COLORMAP_JET)
#     # cv2.imwrite('static/images/perfect binary/'+filename, eroded_image)
    
#     img1 = Image.fromarray(eroded_image).save('wing_segmentation/static/images/perfect binary/'+filename)
#     print("success")

#     img = cv2.resize(eroded_image, (1080, 720), interpolation=cv2.INTER_AREA)
#     img = Image.fromarray(img)

#     # Save the processed image to a BytesIO object
#     img_io = io.BytesIO()
#     img.save(img_io, 'PNG')
#     # img.save('static/images/perfect binary/'+filename, 'PNG')
#     img_io.seek(0)
#     # Return the image as a response

#     return HttpResponse(img_io.getvalue(), content_type='image/png')


# # def process_image(request):
# #     if request.method == 'POST':
# #         image_file = request.FILES.get('image')
# #         erosion_value = int(request.POST.get('erosion', 0))
# #         dilation_value = int(request.POST.get('dilation', 0))

# #         if image_file:
# #             image = Image.open(image_file)
# #             if erosion_value > 0:
# #                 image = image.filter(ImageFilter.MinFilter(size=3))  # Placeholder for erosion
# #             if dilation_value > 0:
# #                 image = image.filter(ImageFilter.MaxFilter(size=3))  # Placeholder for dilation

# #             buffer = io.BytesIO()
# #             image.save(buffer, format='PNG')
# #             buffer.seek(0)
# #             return HttpResponse(buffer.getvalue(), content_type='image/png')
# #         else:
# #             return JsonResponse({'error': 'No image provided'}, status=400)
# #     else:
# #         return render(request, 'upload_image.html')



from django.core.files.base import ContentFile
from django.shortcuts import render
from .models import Wing
from .forms import ImageUploadForm
from io import BytesIO
import base64
from PIL import Image
from PIL import Image, ImageOps

def image_upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = request.FILES['image']
            try:
                fly_id = request.POST['fly_id']
            except:
                fly_id = None

            # Preprocess and process the image
            pre_processed_image = preprocess_image(image, (256, 256))
            processed_image = process_image(pre_processed_image)
            segmented_image = predict(processed_image)
            labeled_image, areas = post_process(segmented_image, 'test.png', (720, 480), 'areas')

            # class_colors = {
            #     0: (0, 0, 0),        # Black for background
            #     1: (255, 0, 0),      # Red
            #     2: (0, 255, 0),      # Green
            #     3: (0, 0, 255),      # Blue
            #     4: (255, 255, 0),    # Yellow
            #     5: (255, 0, 255),    # Magenta
            #     6: (0, 255, 255),    # Cyan
            #     7: (128, 128, 128),  # Gray
            # } 

            # # Create a color image based on the segmentation map
            # height, width = labeled_image.shape
            # color_image = np.zeros((height, width, 3), dtype=np.uint8)

            # for class_index, color in class_colors.items():
            #     color_image[segmented_image == class_index] = color

            # # Convert to PIL Image for further processing or saving
            # color_image_pil = Image.fromarray(color_image)
            
            # color_image_cv = np.array(color_image_pil)

            # # Resize the image using OpenCV
            # resized_image_cv = cv2.resize(color_image_cv, (720, 480), interpolation=cv2.INTER_LANCZOS4)

            # # Convert back to PIL Image if needed
            # resized_image_pil = Image.fromarray(resized_image_cv)

            # Prepare the original image
            buffer_original = BytesIO()
            with Image.open(image) as img:
                img_no_exif = ImageOps.exif_transpose(img)  # Neutralize EXIF-based transformations
                img_gray = img_no_exif.convert('L')        # Convert to grayscale
                img_resized = img_gray.resize((720, 480), Image.Resampling.LANCZOS)
                img_resized.save(buffer_original, format="PNG")

            buffer_original.seek(0)
            original_image_file = ContentFile(buffer_original.read(), name=f"{fly_id}_original.png")
            # original_image_file_f = buffer_original.read(), name=f"{fly_id}_original.png"

            # Prepare the labeled image
            buffer_labeled = BytesIO()
            labeled_image.save(buffer_labeled, format="PNG")
            buffer_labeled.seek(0)
            labeled_image_file = ContentFile(buffer_labeled.read(), name=f"{fly_id}_labeled.png")

            area_2P = areas['2P']
            area_C = areas['C']
            area_M = areas['M']
            area_S = areas['S']
            area_D = areas['D']
            area_1P = areas['1P']
            area_B1 = areas['B1']

            class_colors = {
                0: (0, 0, 0),        # Black for background
                1: (255, 0, 0),      # Red
                2: (0, 255, 0),      # Green
                3: (0, 0, 255),      # Blue
                4: (255, 255, 0),    # Yellow
                5: (255, 0, 255),    # Magenta
                6: (0, 255, 255),    # Cyan
                7: (128, 128, 128),  # Gray
            } 

            # Create a color image based on the segmentation map
            height, width = labeled_image.shape
            color_image = np.zeros((height, width, 3), dtype=np.uint8)

            for class_index, color in class_colors.items():
                color_image[segmented_image == class_index] = color

            # Convert to PIL Image for further processing or saving
            color_image_pil = Image.fromarray(color_image)
            
            color_image_cv = np.array(color_image_pil)

            # Save data to the database
            wing = Wing.objects.create(
                fly_id=fly_id,
                segmented_image=color_image_pil,  # Save labeled image
                original_image=original_image_file,  # Save original image
                area_2P =  area_2P,
                area_C = area_C,
                area_M = area_M,
                area_S = area_S,
                area_D = area_D,
                area_1P =  area_1P,
                area_B1 =  area_B1,
            )
            
            wing.save()

            area_data = [
                {"segment": "2P", "segment_name": "Posterior Region 2", "area": wing.area_2P},
                {"segment": "C", "segment_name": "Central Region", "area": wing.area_C},
                {"segment": "M", "segment_name": "Medial Region", "area": wing.area_M},
                {"segment": "S", "segment_name": "Superior Region", "area": wing.area_S},
                {"segment": "D", "segment_name": "Distal Region", "area": wing.area_D},
                {"segment": "1P", "segment_name": "Posterior Region 1", "area": wing.area_1P},
                {"segment": "B1", "segment_name": "Base Region", "area": wing.area_B1},
            ]

            # Encode images for display 
            buffer_original.seek(0)
            image_original_base64 = base64.b64encode(buffer_original.read()).decode('utf-8')

            buffer_labeled.seek(0)
            image_base64 = base64.b64encode(buffer_labeled.read()).decode('utf-8')
            # image_original_base64 = base64.b64encode(buffer_original.read()).decode('utf-8')
            # image_base64 = base64.b64encode(buffer_labeled.read()).decode('utf-8')

            # Render the template and pass the base64 image string
            return render(request, 'display_image.html', {
                'image_base64': image_base64,
                'og_image': image_original_base64,
                'area_data': area_data
            })
    else:
        form = ImageUploadForm()

    return render(request, 'upload.html', {'form': form})



def home(request):
    if request.method == 'GET':
        return render(request, 'index.html')
    
def login(request):
    if request.method == 'GET':
        return render(request, 'login_user.html')
    
def logout(request):
    if request.method == 'GET':
        return render(request, 'logout.html')
    
def wing_input(request):
    if request.method == 'GET':
        return render(request, 'wing_input.html')
    if request.method == 'POST':
        # image = request.FILES['image']
        image = request.FILES.get('image')
        print('imagesssss', image)
        try:
            fly_id = request.POST.get('fly-id')
        except:
            fly_id = None

        print("IT'S ALIVE");
     
        pre_processed_image = preprocess_image(image, (256, 256))
        processed_image = process_image(pre_processed_image)
        segmented_image = predict(processed_image)
        labeled_image, areas = post_process(segmented_image, 'test.png', (720, 480), 'areas')
     
        buffer_original = BytesIO()
        with Image.open(image) as img:
            img_no_exif = ImageOps.exif_transpose(img)  # Neutralize EXIF-based transformations
            img_gray = img_no_exif.convert('L')        # Convert to grayscale
            img_resized = img_gray.resize((720, 480), Image.Resampling.LANCZOS)
            img_resized.save(buffer_original, format="PNG")
        buffer_original.seek(0)
        original_image_file = ContentFile(buffer_original.read(), name=f"{fly_id}_original.png")

        buffer_labeled = BytesIO()
        labeled_image.save(buffer_labeled, format="PNG")
        buffer_labeled.seek(0)
        labeled_image_file = ContentFile(buffer_labeled.read(), name=f"{fly_id}_labeled.png")


        # Check if a record exists with the given fly_id
        latest_wing = Wing.objects.filter(fly_id=fly_id).order_by('-id').first()


        # Initialize area differences
        area_differences = {}

        

        area_2P = areas['2P']
        area_C = areas['C']
        area_M = areas['M']
        area_S = areas['S']
        area_D = areas['D']
        area_1P = areas['1P']
        area_B1 = areas['B1']
     # Save data to the database
        wing = Wing.objects.create(
            fly_id=fly_id,
            segmented_image=labeled_image_file,  # Save labeled image
            original_image=original_image_file,  # Save original image
            area_2P =  area_2P,
            area_C = area_C,
            area_M = area_M,
            area_S = area_S,
            area_D = area_D,
            area_1P =  area_1P,
            area_B1 =  area_B1,
        )
        
        wing.save()
        if latest_wing:
            area_data = [
                {"segment": "2P", "segment_name": "2nd posterior cell", "area": wing.area_2P, 'difference': latest_wing.area_2P - wing.area_2P},
                {"segment": "C", "segment_name": "Costal cell", "area": wing.area_C, 'difference': latest_wing.area_C - wing.area_C},
                {"segment": "M", "segment_name": "Marginal cell", "area": wing.area_M, 'difference': latest_wing.area_M - wing.area_M},
                {"segment": "S", "segment_name": "Submarginal cell", "area": wing.area_S, 'difference': latest_wing.area_S - wing.area_S},
                {"segment": "D", "segment_name": "Discal cell", "area": wing.area_D, 'difference': latest_wing.area_D - wing.area_D},
                {"segment": "1P", "segment_name": "1st posterior cell", "area": wing.area_1P, 'difference': latest_wing.area_1P - wing.area_1P},
                {"segment": "B1", "segment_name": "Basal cell 1", "area": wing.area_B1, 'difference': latest_wing.area_B1 - wing.area_B1},
            ]
        else:
            area_data = [
                {"segment": "2P", "segment_name": "2nd posterior cell", "area": wing.area_2P},
                {"segment": "C", "segment_name": "Costal cell", "area": wing.area_C},
                {"segment": "M", "segment_name": "Marginal cell", "area": wing.area_M},
                {"segment": "S", "segment_name": "Submarginal cell", "area": wing.area_S},
                {"segment": "D", "segment_name": "Discal cell", "area": wing.area_D},
                {"segment": "1P", "segment_name": "1st posterior cell", "area": wing.area_1P},
                {"segment": "B1", "segment_name": "Basal cell 1", "area": wing.area_B1},
            ]

        if latest_wing:
            print("IT EXISTS")
            # Calculate the difference in area for each segment
            area_differences = [
                {"area": latest_wing.area_2P - wing.area_2P},
                {"area":  latest_wing.area_C - wing.area_C},
                {"area":  latest_wing.area_M - wing.area_M},
                {"area":  latest_wing.area_S - wing.area_S},
                {"area":  latest_wing.area_D - wing.area_D},
                {"area": latest_wing.area_1P - wing.area_1P},
                {"area": latest_wing.area_B1 - wing.area_B1},
            ]
        print(area_differences)
        if len(area_differences) == 0:
            k = False
        else:
            k = True

        print(wing.segmented_image.url);
        # Encode images for display 
        buffer_original.seek(0)
        image_original_base64 = base64.b64encode(buffer_original.read()).decode('utf-8')
        buffer_labeled.seek(0)
        image_base64 = base64.b64encode(buffer_labeled.read()).decode('utf-8')
        # image_original_base64 = base64.b64encode(buffer_original.read()).decode('utf-8')
        # image_base64 = base64.b64encode(buffer_labeled.read()).decode('utf-8')
        # Render the template and pass the base64 image string
        return render(request, 'wing_output.html', {
            'image_url': wing.segmented_image.url,
            'og_image': image_original_base64,
            'area_data': area_data,
            'diff': k
        })
    
def result(request):
    if request.method == 'POST':

        return render(request, 'wing_output.html')
    
def test(request):
    if request.method == 'GET':

        return render(request, 'eye_color_detection_upload.html')
