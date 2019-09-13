#Password Strength Classification with Machine Learning

#imports
import numpy as np
import pandas as pd

#getting the dataset
dataset=pd.read_csv('data.csv',error_bad_lines=False)
dataset.dropna(how='any')  #delete rows that have at least 1 nan
X=dataset.iloc[:,0].values
Y=dataset.iloc[:,1].values

#preprocessing
del_rows=[]
counter=0;
for i in X:
    if(type(i)is not str):
        del_rows.append(counter)
    counter+=1
X=np.delete(X,del_rows)
Y=np.delete(Y,del_rows)
y_del=np.where(np.isnan(Y))
X=np.delete(X,y_del)
Y=np.delete(Y,y_del)

#tokenizing and vectorizing
from keras.preprocessing.text import Tokenizer
tz=Tokenizer(char_level=True,lower=False,filters='')
tz.fit_on_texts(X)
X=tz.texts_to_matrix(X,mode='tfidf')

#splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)

#scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#regression model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,max_iter=1000)
classifier.fit(X_train,Y_train)


#predction using model
Y_pred=classifier.predict(X_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)

#saving the model
import pickle

# Save to file in the current working directory
filename = "password_ML.pkl"  
with open(filename, 'wb') as file:  
    pickle.dump(classifier, file)

'''
# Load from file
with open(pkl_filename, 'rb') as file:  
    pickle_model = pickle.load(file)

# Calculate the accuracy score and predict target values
score = pickle_model.score(X_test, Y_test)  
print("Test score: {0:.2f} %".format(100 * score))  
Ypredict = pickle_model.predict(Xtest) 
'''