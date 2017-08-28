from django import forms
from .models import PostAd
from .models import TabAd
CATEGORIES = (
              ('svmg', 'SVM Gaussian'),
              ('svml', 'SVM Linear'),
              ('ann', 'ANN'),
              ('rfc', 'RFC'),
              )
LOCATIONS = (
             ('uni', 'univariate'),
             ('pca', 'pca'),
             )

class PostAdForm(forms.ModelForm):
    error_css_class = 'error'
    
    #forms.FileField()
    file1 = forms.FileField()
    file2 = forms.FileField()
    
    #category = forms.ChoiceField(choices=CATEGORIES, required=True )
    #location = forms.ChoiceField(choices=LOCATIONS, required=True )
    #description = forms.TextInput()
    
    
    class Meta:
        model = PostAd
        fields = "__all__"
        
        #widgets = {
#    'name': forms.TextInput(attrs={'placeholder': 'What\'s your name?'})#,
        #    'email': forms.TextInput(attrs={'placeholder': 'john@example.com'}),
        #    'gist': forms.TextInput(attrs={'placeholder': 'In a few words, I\'m looking for/to...'}),
#    'expire': forms.TextInput(attrs={'placeholder': 'MM/DD/YYYY'})
#}

class graphForm(forms.ModelForm):
    error_css_class = 'error'
    
    
    
    category = forms.ChoiceField(choices=CATEGORIES, required=True )
    location = forms.ChoiceField(choices=LOCATIONS, required=True )
    #description = forms.TextInput()
    
    
    class Meta:
        model = TabAd
        fields = "__all__"
        
        # widgets = {
#    'name': forms.TextInput(attrs={'placeholder': 'What\'s your name?'})#,
#    'email': forms.TextInput(attrs={'placeholder': 'john@example.com'}),
#    'gist': forms.TextInput(attrs={'placeholder': 'In a few words, I\'m looking for/to...'}),
#    'expire': forms.TextInput(attrs={'placeholder': 'MM/DD/YYYY'})
#}