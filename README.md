# Disease-prediction-using-symptoms

---------------------------Project on Diseases prediction using symptoms------------------------------

The dataset used for this project contains 3 csv files.They are:

    1.Symptom_Description.csv
This file contains different type of diseases(not all that exist in world) in 1st column and a little 
synopsis or a small description about the disease in respective sub-sequence column.

    2.symptom_precaution.csv
This file contains diseases that are mentioned above in 1st column and it also contains some basic 
precautions that are needs to be taken care off in next respective colunm.

    3.dataset.csv
This file contains same diseases mentioned above in 1st column and it also contains different symptoms
possiable in next consecutive columns.

-----------------------------------------Quick Explination---------------------------------------------

1.Basically, like every ML project we also go with cleaning of dataset, fortunately the dataset cleaned
  dataset.So, no need of cleaning it.

2.Next, by checking the dataset itself one can understand that it is categorical and we cannot use it 
  directly to train our model.So, we are creating a new dataset called "TABLE.CSV" using original
  dataset, where columns are different symptoms and output variable is disease.

3.After creating numerical dataset-"TABLE.CSV", we further proceed by splitting dataset to train model
  made on basis of "RandomForestClassifier".

4.On sucessful completion of training and testing of model, we are saving it in "pickle file",inorder
  not to compile model everytime.

5.We are creating another function for predicting disease,precautions and description for "new data 
  point" and returning them and simultaneously adding that datapoint to "TABLE.CSV"

6.Next, getting most common symptoms by setting a threshold value(lets say 400) and asking user to answer "yes/no"
  for those symptoms i.e; weather user is having those symptoms or not.

7.Using these answers we are creating new datapoint, we are passing it as arguments to predictor function and getting the possiable disease, its
  description and few precautions.And displaying it to user.

**NOTE**:
   Run this entire program in terminal or in jupiter/colab notebook only. Because it doesn't contains any GUI, its just a program

The dataset is been taken from kaggle website and here is the link for the dataset
https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset

#software requirements
1. python(>=3.10.2)
2. pandas(>=1.4.2)
3. numpy(>=1.22.2)
4. scikit-learn(>=1.0.2)
(**Note:** This code is written with latest versions)
