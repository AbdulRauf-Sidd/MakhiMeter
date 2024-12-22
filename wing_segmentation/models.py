from django.db import models
import os

class Wing(models.Model):
    fly_id = models.CharField(max_length=50, unique=False, null=True)
    date_added = models.DateTimeField(auto_now_add=True)
    segmented_image = models.ImageField(upload_to='wing_segmentation/static/images/segmented_images/', blank=True, null=True)
    original_image = models.ImageField(upload_to='wing_segmentation/static/images/original_images/', blank=True, null=True)

    # Fields for the 7 segmented areas
    area_2P = models.FloatField(help_text="Area for 2P in square micrometers")
    area_C = models.FloatField(help_text="Area for C in square micrometers")
    area_M = models.FloatField(help_text="Area for M in square micrometers")
    area_S = models.FloatField(help_text="Area for S in square micrometers")
    area_D = models.FloatField(help_text="Area for D in square micrometers")
    area_1P = models.FloatField(help_text="Area for 1P in square micrometers")
    area_B1 = models.FloatField(help_text="Area for B1 in square micrometers")

    # def save(self, *args, **kwargs):
    #     super().save(*args, **kwargs)  # Save to generate the primary key (id)

    #     if self.segmented_image:
    #         original_name = os.path.splitext(self.segmented_image.name)[0]
    #         extension = os.path.splitext(self.segmented_image.name)[1]
    #         new_name = f"{original_name}_Wing_{self.id}{extension}"
    #         self.segmented_image.name = new_name

    #     if self.original_image:
    #         original_name = os.path.splitext(self.original_image.name)[0]
    #         extension = os.path.splitext(self.original_image.name)[1]
    #         new_name = f"{original_name}_Wing_{self.id}{extension}"
    #         self.original_image.name = new_name

    #     super().save(*args, **kwargs)  # Save again with updated file names

    def __str__(self):
        return f"Wing {self.fly_id} added on {self.date_added}"
