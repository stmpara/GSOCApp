# GSOCApp
[GSOC 2017: Web App for Identifying Gene Expression Biomarkers to Classify Cancer Patient Outcomes

* This app is currently in use for lung cancer and 15 genes, but this app will be easily extended to fit all genes and cancer types

This is a Django-based webapp that enables you to 
1. upload csv files in a certain format*
and 
2. produce ROC grpahs about the known 15 feature genes about lung cancer
3. produce AUC graphs based on different feature selection methods and classifiers

This App was produced with the support of Google Summer Of Code and professor Daifeng Wang of Stony Brook University. 

*: Please look at the "uploads" folder in each app to see what formats of csv meet the requirement. 

This repository has two projects inside: gsocApp and gsocApplight.
While gsocApp is only available for one classifier(support vector machine gaussian kernel), goscApplight is available for three other 
classifiers as well: support vector machine linear kernel, artificial neural network, random forest.
Due to the heavy running time and large size of data, it is recommended that you use gsocApp for testing and deplyoing purposes.


## Getting Started

(Assuming that you have Django and Python3 installed)
Download the gsocApp folder.
Go to the directory with manage.py.
Now, run 
```
python3 manage.py makemigrations
```

```
python3 manage.py migrate
```

Finally,

```
python3 manage.py runserver
```

and the program is up and running at http://127.0.0.1:8000.

Now, please go to 
```
http://127.0.0.1:8000/postad 
```
at your browser, and read on to the next section.


## Uploading Files

![Alt text](/ScreenShots/file_upload.png)

You will see the above view when you access 

```
http://127.0.0.1:8000/postad 
```

To upload files correctly, go to folder sample_csv and upload LuncClinical.csv as the 1st File.
For the second file, go to 
```
https://drive.google.com/file/d/0Bz05ZzbJ7bR_QWUwRTJIeUprWmc/view?usp=sharing
```
and download from the link. Now upload this gene expression file as the second file.


Now, wait for 10 minutes until you see the following screen:

![Alt text](/ScreenShots/graph_button.png)


## Drawing Graphs

Once you see the view above with dropdowns and draw graph button, select the classifier and feature extraction method you'd like.
Wait for another 5 minutes after pressing the button.
You will see the following:

![Alt text](/ScreenShots/1st_tab.png)
![Alt text](/ScreenShots/2nd_tab.png)
![Alt text](/ScreenShots/3rd_tab.png)
![Alt text](/ScreenShots/4th_tab.png)


## Additional Works to Be Done
There are a few unfinished works in this project.
They are 
1. While the original purpose of the project was to provide personalized drug matching, the uploaded code does not contain anything about this.
2. The number of features selected is different from time to time.
I think this is because, due to the fact that it is too time consuming to calculate the AUC every time for all of 1~50 features, 
I used 1 - E[tpr] E[1 - fpr] as the estimator for AUC, where E[tpr] is the mean of true positive rates.
(I referred to here: https://github.com/fchollet/keras/issues/1732)
I would appreciate any feedback about the right usage of this estimator and why this bug occurs. 


### Prerequisites

You need the following prerequisites to run the software.
```
(1) Python3
(2) Numpy
(3) SKLearn
(4) Django
(5) Scipy
(6) Matplotlib
(7) Bokeh
(8) Tensorflow
(9) Keras with Tensorflow Backend
(10) Pandas
```

Please use the latest versions when installing the above. It is recommended that you install the latest Anaconda for python 3 since it automatically installs many of the above.
Python3 Anaconda: https://www.anaconda.com/download/

## Deployment

If you would like to deploy this on a live system with an actual non-local url, you can easily do so. 
Add additional notes about how to deploy this on a live system, please look at "Deploying Django" at the official Django documentation.

## Built With

* [Scikitlearn] (http://scikit-learn.org/stable/) - Python Machine Learning Framework
* [Tensorflow] (https://www.tensorflow.org/) - Open Source Library for Machine Intelligence
* [Bokeh] (https://bokeh.pydata.org/en/latest/) - Python-based Web Data Visualization
* [Django] (https://www.djangoproject.com/) - The Web Framework Used in This Project

## Authors

* **So Yeon Min** - *Initial work* - [TiffanyMin](https://github.com/TiffanyMin)



## Acknowledgments

This App was produced with the support of Google Summer Of Code and professor Daifeng Wang of Stony Brook University. 
