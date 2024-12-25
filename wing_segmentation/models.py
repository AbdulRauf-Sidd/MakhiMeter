from django.db import models
import os

class Wing(models.Model):
    fly_id = models.CharField(max_length=50, unique=False, null=True)
    date_added = models.DateTimeField(auto_now_add=True)
    segmented_image = models.ImageField(upload_to='wing_segmentation/static/images/segmented_images/', blank=True, null=True)
    original_image = models.ImageField(upload_to='wing_segmentation/static/images/original_images/', blank=True, null=True)

    # Fields for the 7 segmented areas
    area_2P = models.FloatField(help_text="Area for 2P in square micrometers", default=0)
    area_3P = models.FloatField(help_text="Area for 3P in square micrometers", default=0)
    area_M = models.FloatField(help_text="Area for M in square micrometers", default=0)
    area_S = models.FloatField(help_text="Area for S in square micrometers", default=0)
    area_D = models.FloatField(help_text="Area for D in square micrometers", default=0)
    area_1P = models.FloatField(help_text="Area for 1P in square micrometers", default=0)
    area_B1 = models.FloatField(help_text="Area for B1 in square micrometers", default=0)

    def __str__(self):
        return f"Wing {self.fly_id} added on {self.date_added}"
