from .helpers import *
from django.shortcuts import render, redirect
from .models import Wing
from django.core.files.base import ContentFile
from .models import Wing
from io import BytesIO
from django.contrib.auth import authenticate, login, logout
from PIL import Image, ImageOps
from django.http import HttpResponse
from reportlab.pdfgen import canvas
from django.shortcuts import get_object_or_404
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from .forms import CustomUserCreationForm
import os
from django.contrib import messages


def home(request):
    if request.method == 'GET':
        return render(request, 'index.html')

def logout_view(request):
    # Logout the user
    logout(request)
    print(request.user)
    
    # Redirect to the login page or homepage after logout
    return redirect('/')

def login_view(request):
    if request.method == 'GET':
        return render(request, 'login_user.html')
    elif request.method == 'POST':
        # Get username and password from the form
        username = request.POST['username']
        password = request.POST['password']

        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # If authentication is successful, log the user in
            login(request, user)
            messages.success(request, "You have logged in successfully!")
            return redirect('/')  # Redirect to the homepage or a dashboard page
        else:
            # If authentication fails, show an error message
            messages.error(request, "Invalid username or password.")
            return redirect('login_user')  # Redirect back to login page
    
def register(request):
    if request.method == 'GET':
        form = CustomUserCreationForm()
        return render(request, 'register.html', {'form': form})
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    
# def logout(request):
#     if request.method == 'GET':
#         return render(request, 'logout.html')
    
def wing_upload(request):
    if request.method == 'GET':
        return render(request, 'wing_segmentation_upload.html') 
    if request.method == 'POST':
        
        size = (1250, 1950)
        image = request.FILES.get('img')
        
        buffer_original = BytesIO()
        with Image.open(image) as img:
            width, height = img.size
            img_no_exif = ImageOps.exif_transpose(img)  # Neutralize EXIF-based transformations
            img_gray = img_no_exif.convert('L')        # Convert to grayscale
            img_resized = img_gray.resize(size, Image.Resampling.LANCZOS)
            img_resized.save(buffer_original, format="PNG")
        buffer_original.seek(0)
        original_image_file = ContentFile(buffer_original.read(), name=f"original.png")
        
        if width > height:
            size = (size[1], size[0])


        pre_processed_image = preprocess_image(image, (256, 256))
        processed_image = process_image(pre_processed_image)
        segmented_image = predict(processed_image)
        labeled_image, areas, seg = post_process(segmented_image, image, size)

    
        buffer_labeled = BytesIO()
        labeled_image.save(buffer_labeled, format="PNG")
        buffer_labeled.seek(0)
        labeled_image_file = ContentFile(buffer_labeled.read(), name=f"labeled.png")


        # with Image.open(seg) as img:
        #     img = ImageOps.exif_transpose(img)
        #     buffer_labeled = BytesIO()
        #     img.save(buffer_labeled, format="PNG")
        #     buffer_labeled.seek(0)
        #     labeled_image_file = ContentFile(buffer_labeled.read(), name=f"labeled.png")
        
        # labeled_image_file = ContentFile(buffer_labeled.read(), name=f"{fly_id}_labeled.png")
        # original_image_file = ContentFile(buffer_original.read(), name=f"{fly_id}_original.png")

    
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
            'wing_id': wing.id,
        })
    

def download_results(request, wing_id):
    # Get the Wing object or return 404 if not found
    wing = get_object_or_404(Wing, id=wing_id)

    # Prepare PDF response
    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename=wing_results_{wing.fly_id}.pdf'

    # Create PDF canvas
    buffer = canvas.Canvas(response, pagesize=A4)
    width, height = A4

    # Title
    buffer.setFont("Helvetica-Bold", 20)
    buffer.drawString(150, height - 100, f"Wing Segmentation Report")

    # Load original image
    if wing.original_image:
        image_path = wing.original_image.path
        if os.path.exists(image_path):
            img = ImageReader(image_path)
            buffer.drawImage(img, 100, height - 500, width=400, height=300)

    # Draw table with segmented areas
    buffer.setFont("Helvetica", 14)
    buffer.drawString(100, height - 520, "Segmented Areas (µm²):")

    area_data = {
        '2P': wing.area_2P,
        '3P': wing.area_3P,
        'M': wing.area_M,
        'S': wing.area_S,
        'D': wing.area_D,
        '1P': wing.area_1P,
        'B1': wing.area_B1
    }

    y_offset = height - 560
    row_height = 20

    for segment, area in area_data.items():
        y_offset -= row_height
        buffer.drawString(100, y_offset, f"{segment}: {area} µm²")

    buffer.showPage()
    buffer.save()

    return response


# def test(request):
#     if request.method == 'POST' and request.FILES.get('image'):
#         uploaded_image = request.FILES['image']

#         uploaded_image
#         # Process the image
#         processed_image = testing(uploaded_image)
        
#         # Save processed image to an in-memory byte stream
#         image_io = BytesIO()
#         processed_image.save(image_io, format='JPEG')
#         image_io.seek(0)

#         # Return the processed image as an HTTP response
#         return HttpResponse(image_io, content_type='image/jpeg')

#     return render(request, 'test.html')