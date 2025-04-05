from django.shortcuts import render

# Create your views here.

# Create your views here.
def gender_upload(request):
    if request.method == 'GET':
        return render(request, 'gender_identification_upload.html')
    
def gender_output(request):
    if request.method == 'GET':
        context = {
            'url': '/static/images/flight_output_image.jpg'
        }
        return render(request, 'gender_identification_output.html', context)