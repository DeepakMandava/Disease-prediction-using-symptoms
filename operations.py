def compile_model():
    #importing required modules and libraries.
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
    import joblib

    #loading datasets
    dataset = pd.read_csv('data\dataset.csv')
    description = pd.read_csv('data\symptom_Description.csv')
    precautions = pd.read_csv('data\symptom_precaution.csv')

    '''
    #checking attributes type, and NAN values
    print(dataset.info())
    
    # viewing dataset
    print(dataset.head(5))
    print(description.head(5))
    print(precautions.head(5))
    '''
    
#Pre-processing the dataset

    # adding "symptoms" column to store all the symptoms at one place
    dataset["symptoms"] = 0
    
    #adding all the symptoms to "symptoms" column as a list
    records = dataset.shape[0]
    for i in range(records):
        values = dataset.iloc[i].values
        values = values.tolist()
        if 0 in values:
            dataset["symptoms"][i] = values[1:values.index(0)]
        else:
            dataset["symptoms"][i] = values[1:]
    
    '''
    Now let us create a dataset named "table" in such a way that,
    columns are names with symptoms, output variable as disease.
    This will ease our work for classification
    '''
    
    column_names = dataset[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4',
                        'Symptom_5', 'Symptom_6', 'Symptom_7', 'Symptom_8', 'Symptom_9',
                        'Symptom_10', 'Symptom_11', 'Symptom_12', 'Symptom_13', 'Symptom_14',
                        'Symptom_15', 'Symptom_16', 'Symptom_17']].values.ravel()
    
    """Here in the above code, **dataset[['symptom_1','symptom_2',......,'symptom_17']].values** returns an **2-D array** with values of each symptom as row.
    
    And **reval() function converts 2D/multidimentional array into flatten-continous array**.To learn more about reval() [click here](https://www.javatpoint.com/numpy-ravel#:~:text=%20numpy.ravel%20%28%29%20in%20Python.%20The%20numpy%20module,type%20as%20the%20source%20array%20or%20input%20array.)
    """
    
    '''
    checking for any repetition of symptoms and avoiding them. 
    Because, they cause increase in computational time
    '''
    
    column_names = pd.unique(column_names)
    column_names = column_names.tolist()
    column_names = [i for i in column_names if str(i) != "nan"]
    
    # Now, we have colunm_names, lets proceed and create "table"
    table = pd.DataFrame(columns=column_names,index = dataset.index)
    table["symptoms"] = dataset["symptoms"]
    for i in column_names:
        for j in range(table.shape[0]):
            if(i in table['symptoms'][j] and i!="nan"):
              table[i][j]=1
            else:
              table[i][j]=0
    
    # Now lets add disease column to "table"
    table['disease'] = dataset['Disease']
    
    # Also remove symptoms column from "table"
    table.drop('symptoms',inplace=True,axis=1)
    
    '''
    # Look at final "table"
    table.head(5)
    '''
    
    """# **Creating model using RandomForestClassifier**"""
    
    # Splitting "table" into dependent(y) and independent(x) variables
    x = table.iloc[:,:131]
    y = table.iloc[:,131]
    
    # Splitting x,y for training and testing
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,shuffle=True)
    
    # creating classifier using RandomForestClassifier
    classifier = RandomForestClassifier()
    
    # Training the classifier with training data
    classifier.fit(x_train,y_train)
    
    # predicting disease for testing data
    y_pred = classifier.predict(x_test)
    
    # checking accuracy of classifier
    accuracy_score(y_test,y_pred)
    
    # confusion matrix of classifier
    confusion_matrix(y_test,y_pred)
    
    # saving model as "disease_predictor.pkl"(a pickle file)
    joblib.dump(classifier, 'disease_predictor.pkl')
    table.to_csv('table.csv',index=False)

    print("Random Forest Classifier model has been sucessfully created and pickle file is saved along with table-dataset")




# checking for prediction of new data
def predictor(new_data):
    import joblib
    import pandas as pd

    if isinstance(new_data,list) and len(new_data)==132:    
        #loading required datasets
        description = pd.read_csv('data\symptom_Description.csv')
        precautions = pd.read_csv('data\symptom_precaution.csv')
        table = pd.read_csv('table.csv')
    

        #This new_data is related to "disease-fungal infection"
        table.loc[len(table.index),:] = new_data
        
        #loading ML-classifier that is saved
        classifier = joblib.load('disease_predictor.pkl')
    
        # Trying to predict disease of latest data that entered in "table"
        pred = classifier.predict(table.iloc[len(table.index)-1:len(table.index),:131])
        
        # This is to delete the last entered datapoint incase if you dont want to edit dataset, uncomment next line
        #table.drop(len(table.index)-1,axis=0,inplace=True)
    
        #adding disease of new data to its corresponding disease column as it is set to '0' before
        table.iloc[len(table.index)-1,131] = pred[0]
    
        #again saving "table" dataset to .csv file under folder-data
        table.to_csv('table.csv',index=False)
        
        # getting description about predicted disease
        desc = description.loc[description['Disease'].values==pred[0],'Description'].iloc[0]
    
        # Getting  precautions about predicted disease
        prec = precautions.loc[precautions['Disease']==pred[0],:]
    
        # returing the predicted disease
        return [pred[0],desc,prec]
    
    else:
        print('New data entered is either insufficient or datatype is mismatched')
        return 0
'''
Import this file to your script(in which you want to work), then you can use the functions to complete the task 
'''
