from django.db import models

class Passenger(models.Model):
    name = models.CharField(max_length=100)
    pclass = models.IntegerField()
    sex = models.CharField(max_length=10)
    age = models.FloatField()
    fare = models.FloatField()
    family_size = models.IntegerField()
    embarked_q = models.IntegerField()
    embarked_s = models.IntegerField()
    survived = models.BooleanField()
