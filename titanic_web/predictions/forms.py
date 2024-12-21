from django import forms

class PassengerForm(forms.Form):
    name = forms.CharField(max_length=100)
    pclass = forms.IntegerField()
    sex = forms.ChoiceField(choices=[('male', 'Male'), ('female', 'Female')])
    age = forms.FloatField()
    fare = forms.FloatField()
    family_size = forms.IntegerField()
    embarked_q = forms.IntegerField()
    embarked_s = forms.IntegerField()
