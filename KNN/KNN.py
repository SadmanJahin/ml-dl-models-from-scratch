import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from collections import Counter

def knnAlgorithm(dataset,prediction,neighbours):
    row=len(dataset)
    result=[]*row
    for x in range(row):
        speed= dataset.loc[x, 'speed']
        analysis= dataset.loc[x, 'analysis']
        result.append(math.sqrt(pow(prediction[0]-speed, 2)+pow(prediction[1]-analysis, 2)))
    temp = result[:]
    index=[]
    result.sort()
    for count,value in enumerate(result):
        for idx,check in enumerate(temp):
            if(check == value):
                index.append(idx)
        if(count==neighbours-1):
            break
    predictedrisk=[]
    for x in index:
        predictedrisk.append(dataset.loc[x, 'risk'])
    mode = Counter(predictedrisk)
    mode.most_common(1)

    if mode.most_common(1)[0][0]=='0' : return "Low"
    elif mode.most_common(1)[0][0]=='1' : return "Medium"
    else : return "High"





car_data=pd.read_csv('car driving risk analysis  modified.csv')
encoded_dataset=car_data
row=len(encoded_dataset)
for x in range(row):
 if(encoded_dataset.loc[x, 'risk'] == 'high'): encoded_dataset.loc[x, 'risk'] = '2'
 elif(encoded_dataset.loc[x, 'risk'] == 'medium'): encoded_dataset.loc[x, 'risk'] = '1'
 else : encoded_dataset.loc[x, 'risk'] = '0'

neighbours=int(input("Neighbours?: "))
speed, analysis = input("Enter Speed and Analysis for prediction : ").split()
speed=int(speed)
analysis=int(analysis)
prediction=knnAlgorithm(encoded_dataset,[speed,analysis],neighbours)
print("Prediction for Speed:"+str(speed)+" and Analysis: "+str(analysis)+" is Risk :"+str(prediction))

