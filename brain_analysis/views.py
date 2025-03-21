from django.shortcuts import render

# Create your views here.
def brain_home(request):
    if request.method == 'GET':
        return render(request, 'brain.html')
    

def brain_output(request):
    if request.method == 'GET':
        context = {
        'full_area': 402.5,
        'm_area': 100.2,
        'r': (0.42),
        'image_url': '/static/images/top right.png'
        }
        return render(request, 'brain_output.html', context)