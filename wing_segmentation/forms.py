from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField()


class ImageUploadForm2(forms.Form):
    image = forms.ImageField()

    
