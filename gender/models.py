from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Gender(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True)
    image = models.ImageField(upload_to='gender/original/', null=True)
    classification = models.CharField(max_length=10, null=True)
    hog = models.ImageField(upload_to='gender/hog/')

    def __str__(self):
        return self.name