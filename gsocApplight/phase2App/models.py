from django.db import models
from .backend import roc, utils, connection
from functools import lru_cache

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

class PostAd(models.Model):
    #upload_to='uploads/',
    #upload_to='uploads/',
    # models.FileField( default = '/uploads/LuncClinical1.csv')
    file1 = models.FileField(upload_to='uploads/', default = '/uploads/LuncClinical.csv')
    file2 = models.FileField(upload_to='uploads/', default = '/uploads/15_gene_file.xlsx_tab1.csv')
    #_list1 = None
    #self.X, self.Y, self.X_train, self.X_test, self.y_train, self.y_test, self.geneIDList, self.DescList = None, None, None, None, None, None, None, None
    
    #name        = models.CharField(max_length=50)
    #email       = models.EmailField()
    #gist        = models.CharField(max_length=50)
    #category    = models.CharField(max_length=500,null=True, choices=CATEGORIES, default = 'svmg')
#location    = models.CharField(max_length=500,null=True, choices=LOCATIONS, default = 'uni')
    #description = models.TextField(max_length=300)
    #expire      = models.DateField()
    
    @property
    @lru_cache(maxsize=None)
    def get_computed(self):
        return utils.getDataAll('uploads/'+ str(self.file1.name), 'uploads/'+str(self.file2.name))

    @property
    @lru_cache(maxsize=None)
    def get_fifteen(self):
        return utils.getData('uploads/'+ str(self.file1.name), 'uploads/'+'15_gene_file.xlsx_tab1.csv')

    #return self._list1
    #return [X, self.Y, self.X_train, self.X_test, self.y_train, self.y_test, self.geneIDList, self.DescList]

#@get_computed.setter
#def set_computed(self):
#    self._list1 = utils.getDataAll(self.file1, self.file2)

#list1 = property(get_computed, set_computed)

#class XYs():
#    def __init__(self, file1, file2):
#        self.X, self.Y, self.X_train, self.X_test, self.y_train, self.y_test, self.geneIDList, self.DescList = utils.getDataAll(file1, file2)
#        self.file1, self.file2 = self.file1, self.file2


class TabAd(models.Model):
    category    = models.CharField(max_length=500, choices=CATEGORIES, null=True, default = 'svmg')
    location    = models.CharField(max_length=500, choices=LOCATIONS, null= True, default = 'uni')