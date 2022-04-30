from operations import predictor
from os import path
import pandas as pd
import numpy as np

'''
checking if model is already exist, if yes, then we are directly predicting for new data point
else, we are first compiling the model and then we are predicting the disease for  new data point
'''

def collect_data(threshold):
    # reading "table" dataset use and to add new datapoints to it
    table = pd.read_csv('table.csv')

    # making copy of symptoms and arranging symptoms in descending order with a threshold value as lower limit
    s = table.iloc[:,:131].sum().sort_values(ascending=False)[table.iloc[:,:131].sum()>float(threshold)]

    # creating new data_point from symptoms
    print('Type "yes" if you have the mentioned symptom, else type "no"\n')
    new_data = [0.0 for i in range(132)]
    count=0
    for row in s.index:
        count = count+1
        data_index = 0
        if(count!=1):
            print(count-1,". ",row,"?\n")
            answer = input()
            print("\n")
            if answer=='yes':
                for i in table.columns:
                    if i==row:
                        new_data[data_index]=1.0
                    data_index = data_index+1
    return new_data


def output(results):
    # disease will be displayed
    print("Suspected disease: ",results[0].upper(),"\n")

    # description will be displayed
    print("Small description:\n",results[1],"\n")

    # precautions will be displayed
    print("Suggestions/precautions:")
    for i in range(1,5):
        print(results[2].iloc[0,i],"\n")
    
    return 0

if(path.exists('disease_predictor.pkl')):
    
    new_data = collect_data(700)

    # new datapoint is created from symptoms
    results = predictor(new_data)

    #displaying the output
    output(results)

else:
    new_data = collect_data(500)
    
    # new datapoint is created from symptoms
    results = predictor(new_data)

    #displaying the output
    output(results)
