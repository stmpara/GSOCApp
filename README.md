# GSOCApp
[GSOC 2017: Web Framework for Lung Cancer Data ROC Visualization]
This is a Django-based webapp that enables you to 
(1) upload csv files in a certain format*
and 
(2) produce ROC grpahs about the known 15 feature genes about lung cancer
(3) produce AUC graphs based on different feature selection methods and classifiers

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
(1) python3 manage.py makemigrations
(2) python3 manage.py migrate

Finally,

python3 manage.py runserver

and the program is up and running at http://127.0.0.1:8000.

Now, please go to http://127.0.0.1:8000/postad at your browser, and upload files.
The first file has to be the lung clinical file, the second has to be the gene expression csv(as in the /uploads/ folder).
Upload, wait for 5 minutes, and then you are prompted to draw graphs!


### Prerequisites

You need the following prerequisites to run the software.
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

Please use the latest versions when installing the above. It is recommended that you install the latest Anaconda for python 3 since it automatically installs many of the above.
Python3 Anaconda: https://www.anaconda.com/download/

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

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
