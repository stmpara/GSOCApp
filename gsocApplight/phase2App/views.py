from django.http import HttpResponse, HttpResponseRedirect
from django.http import Http404
from django.shortcuts import render, render_to_response
from django.views.generic import FormView
from bokeh.plotting import figure, output_file, show
#from bokeh.io import  vform
from bokeh.resources import CDN
from bokeh.embed import components
from bokeh.models import ColumnDataSource, NumberFormatter
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.widgets import DataTable, TableColumn
from bokeh.palettes import Spectral11


import pandas as pd


import math
from django.core.urlresolvers import reverse

from keras.utils import np_utils

from .models import PostAd, TabAd
from .forms import PostAdForm
from .forms import graphForm

from .backend import roc, utils, connection


from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

class PostAdPage(FormView):
    template_name = 'post_ad.html'
    success_url = '/awesome/'
    form_class = PostAdForm
    
    #if form is valid
    def form_valid(self, form):
            # process form data
        obj = PostAd() #gets new object
        #obj.name = form.cleaned_data['name']
        obj.file1 = form.cleaned_data['file1']
        obj.file2 = form.cleaned_data['file2']
        #obj.category = 'svmg'
        #obj.location = 'uni'
        #obj.category = form.cleaned_data['category']
        #obj.location = form.cleaned_data['location']
        # obj.description = form.cleaned_data['description']
        #obj.expire = form.cleaned_data['expire']
        #finally save the object in db
        obj.save()
        gform =graphForm()
        #Can change this to graph
        #return HttpResponse("Sweeeeeet.")
        ##Should define OBJ again but anyways
        #return render_to_response('index.html', {'form': gform})
        return HttpResponseRedirect(reverse('index'))


def index(request):
    num_classes = 4
    n_components = 13
    #form = graphForm()
    try:
        obj = PostAd.objects.latest('id')
        
        graphObj = TabAd()
    
    except PostAd.DoesNotExist:
        raise Http404('This item does not exist')
    
    
    
    ###Phase2LightApp
    
    [X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList] = obj.get_computed
    [X1, Y1, X_train1, X_test1, y_train1, y_test1, geneIDList1, DescList1] = obj.get_fifteen

    y_trainann = np_utils.to_categorical(y_train,num_classes)
    y_testann = np_utils.to_categorical(y_test, num_classes)
        
        
    cl1 = svm.SVC(kernel='rbf', probability=True)
    cl2 = svm.SVC(kernel='linear', probability=True)
    n_components = X_train.shape[1]
    if n_components <= 15:
        cl3 = roc.initialize_nn(n_components)
    else:
        cl3 = roc.initialize_nnAll(n_components)
    cl4 = RandomForestClassifier(n_estimators = 50)
        
    clList = [cl1, cl2, cl3, cl4]
        
    clTickerList = ["svmg"]
    selTickerList = ["pca", "uni"]
    ROCDict={}
    TableDict={}


    for i in range(1):
        acl = clList[i]
        tick = clTickerList[i]
        TableDict[tick] = {}
        ROCDict[tick] = {}
        for j in selTickerList:
            if j == "ann":
                TableDict[tick][j], ROCDict[tick][j] = connection.intoDFFromStart(X, Y, X_train, X_test, y_trainann, y_testann, geneIDList, DescList, acl, j)
            else:
                TableDict[tick][j], ROCDict[tick][j] = connection.intoDFFromStart(X, Y, X_train, X_test, y_train, y_test, geneIDList, DescList, acl, j)


    ROC15Dict = {"svmg":roc.plotROC_SVMGaussian(X_train1, X_test1, y_train1, y_test1)
        #,"svml":roc.plotROC_SVMLinear(X_train, X_test, y_train, y_test),
        #"ann":roc.mainPlotFunc(X_train, X_test, y_trainann, y_testann),
        #"rfc":roc.plotROC_RFC(X_train, X_test, y_train, y_test)
    }

    df3 = utils.intoDataFrame(X1)
    xColumns = columns=["AKT1", "ALK", "BRAF", "DDR2", "EGFR", "FGFR1", "KRAS", "MET", "NRAS", "PIK3CA", "PTEN", "RET", "ROS1"]
    df3columns = [TableColumn(field=str(a), title=str(a)) for a in xColumns]
    data_table3 = DataTable(source=ColumnDataSource(df3), columns=df3columns)



    if request.method == "GET" :
        
        #return render_to_response('index.html', {'form': form})
        return render_to_response('index.html')
    
    elif request.method == "POST" :
        ##So these come from the html
        #if form.is_valid():
            #print('form is valid')
            #print(form.cleaned_data)
        graphObj.category =request.POST['category']
        graphObj.location =request.POST['locations']
        graphObj.save()
        
        #########다음 할 것:
        #########ROC의 Matplotlib을 x, y로 저장하도록 바꾸기
        #########그래서 ROC DICT도 ROCDict[svmg][uni] = (x,y) 일케 되게
        
        cat =graphObj.category
        loc =graphObj.location
        
        realx = ROCDict[cat][loc][0]
        realy = ROCDict[cat][loc][1]
        
        
        #x15 = ROC15Dict[cat][0]
        #y15 =ROC15Dict[cat][1]
        
        x15 = []
        y15 = []
        [fpr,tpr, auc] = ROC15Dict[cat]
        x15.append(fpr["micro"])
        y15.append(tpr["micro"])
        x15.append(fpr["macro"])
        y15.append(tpr["macro"])
        for i in range(num_classes):
            x15.append(fpr[i]), y15.append(fpr[i])



        df = TableDict[cat][loc]
        
        #domain  = request.POST['domain'].split()
        #eqn     = request.POST['equation']
        #domain = range( int(domain[0]), int(domain[1]) )
        #y = [ eval(eqn) for x in domain ]
        title = 'y = '
        
        plot1 = figure(title= title , x_axis_label= 'Number of Features(genes)', y_axis_label= 'AUC Score', plot_width =500, plot_height =500)
        #tabs = Tabs(tabs=[ tab1 ])
        
        ##Just need to change here: domain, y
        #plot.line(domain, y, legend= 'f(x)', line_width = 2)
        plot1.line(realx, realy, legend= 'f(x)', line_width = 2)
        tab1 = Panel(child=plot1, title="AUC of the best features")
        
        
        #plot2 = figure(title= title , x_axis_label= 'False Positive Rate', y_axis_label= 'True Postive Rate', plot_width =400, plot_height =400)
        #plot2.line(x15, y15, legend= 'f(x)', line_width = 2)
        #tab2 =Panel(child=plot2, title="ROC of the best features")

        #data = dict(
        #            dates=[date(2014, 3, i+1) for i in range(10)],
        #            downloads=[randint(0, 100) for i in range(10)],
        #            )
        #source = ColumnDataSource(df)

        #columns = [
        #   TableColumn(field="dates", title="Date", formatter=DateFormatter()),
        #   TableColumn(field="downloads", title="Downloads"),
        #   ]
        #data_table = DataTable(source=source, columns=columns, width=400, height=280)
        #data_table = DataTable(source=source, width=400, height=10000)

        #'Feature Genes': columns,
        #    'Descriptions': desc,
        #'Feature Importance': FeatureImp
        
        
        #df2 = pd.DataFrame.from_csv(obj.file1)
        #dictionary = {"name":["1.00","2.00"],
        #                "salary":["5", "6"]}
        
        
        #df2 =pd.DataFrame.from_dict(dictionary)
        df2 =TableDict[cat][loc]
        #df3 =pd.DataFrame.from_csv(obj.file2)
        
        source2 =ColumnDataSource(df2)
        #source3 =ColumnDataSource(df3)

        
        #data_table2 = DataTable(source=source2)
        
        columns = [
                   TableColumn(field='Feature Genes', title="Feature Genes"),
                   TableColumn(field='Descriptions', title="Descriptions"),
                   TableColumn(field='Feature Impotance', title="Feature Impotance")
                   ]

        data_table2 = DataTable(source=source2, columns=columns)
        




        ###For the CSV's
        #csvDF1 = pd.DataFrame.from_csv('uploads/'+ str(obj.file1.name))
        
        ##csvDF2 = pd.DataFrame.from_csv('uploads/'+str(obj.file2.name))
        #dfHeader1= list(csvDF1)
        #columnsCSV1 = [TableColumn(field = a) for a in dfHeader1]
        
        #data_table3 = DataTable(source=ColumnDataSource(csvDF1), columns=columnsCSV1)
        #tab4 =Panel(child=data_table3, title="Lung Clinical Table")
        
        mypalette=Spectral11[0:num_classes]
        
        
        #data_table3 = DataTable(source=source3, width=400000, height=10000)
        plot2 = figure(title= title , x_axis_label= 'X-Axis', y_axis_label= 'Y- Axis', plot_width =800, plot_height =800)

        plot2.line(fpr["micro"], tpr["micro"], line_width=2, line_color = 'red', legend = 'micro-average ROC curve (area = {0:0.2f})'''.format(auc["micro"]))
        plot2.line(fpr["macro"], tpr["macro"], line_width=2, line_color = 'yellow', legend = 'macro-average ROC curve (area = {0:0.2f})'''.format(auc["macro"]))

        for i, color in zip(range(num_classes), mypalette):
            plot2.line(fpr[i], tpr[i], line_width=2, line_color =color , legend = 'ROC curve of class {0} (area = {1:0.2f})'''.format(i, auc[i]))
        #plot2.multi_line(x15, y15, line_width=2, line_color=mypalette,)
        #Maybe to vform here
        tab2 =Panel(child=plot2, title="15 Known Genes ROC")
        tab3 = Panel(child=data_table2, title="Best Features Table")
        #tab4 =Panel(child=data_table3, title="clinical Table")
        tab4 = Panel(child = data_table3, title = "15 Genes Table")
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4 ])
        script, div = components(tabs)
        #script, div = components(plot1)
        
        #return render_to_response( 'index.html', {'script' : script , 'div' : div, 'form': form}, )
        return render_to_response( 'index.html', {'script' : script , 'div' : div}, )
    
    
    else:
        pass


#def index(request):
#    if request.method == "GET" :
#        return render_to_response('index.html')
#
#    elif request.method == "POST" :
#        ##So these come from the html
#        domain  = request.POST['domain'].split()
#        eqn     = request.POST['equation']
#        domain = range( int(domain[0]), int(domain[1]) )
#        y = [ eval(eqn) for x in domain ]
#        title = 'y = ' + eqn
#
#        plot = figure(title= title , x_axis_label= 'X-Axis', y_axis_label= 'Y- Axis', plot_width =400, plot_height =400)
#        plot.line(domain, y, legend= 'f(x)', line_width = 2)
#        script, div = components(plot)
        
        #        return render_to_response( 'index.html', {'script' : script , 'div' : div}, )


#    else:
#        pass