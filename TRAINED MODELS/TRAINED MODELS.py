#!/usr/bin/env python
# coding: utf-8

# ## This Code contains trained models.

import tensorflow
import sklearn
import numpy as np
from sklearn.externals import joblib 

DNN = joblib.load('DNN.pkl')
LR = joblib.load('Linear_Regression.pkl')
SVR = joblib.load('SVR.pkl')


# ### Add Your Data You Want To Predict. 
#    Make Sure to input the data in shape of (6,x), here x is the number of test samples
#    
#    
#    ##### for example: Look Below


b=np.array([[ 0.07412601,  0.84832445, -0.48999052,  1.01866265,  0.12999732,
        1.10757154]])

DNN.predict(b)

