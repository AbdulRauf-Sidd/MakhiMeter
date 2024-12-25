from django.shortcuts import render

# Create your views here.
def brain_home(request):
    if request.method == 'GET':
        return render(request, 'brain.html')