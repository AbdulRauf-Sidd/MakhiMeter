from django.shortcuts import render


# Create your views here.
def flight_upload(request):
    if request.method == 'GET':
        return render(request, 'flight_input.html')
    
def flight_output(request):
    if request.method == 'GET':
        context = {
            'url': '/static/images/flight_output_image.jpg'
        }
        return render(request, 'flight_output.html', context)