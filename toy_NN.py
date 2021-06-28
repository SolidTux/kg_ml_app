
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import sys
import numpy as np
import pandas as pd

#constants
RS = 42  #a random state
n_headers = ['lgE', 'X', 'Y', 'CoreDist', 'Ze', 'Az', 'lgNe', 'lgNmu', 'Age']    #for binary classification

def train_nn():
    #-------------------------------------------------------------------------------------------
    #                Pt.1. Data preparation and pre-processing
    #-------------------------------------------------------------------------------------------

    #create a fake dataset
    x, y = make_classification(
            n_samples=200,
            n_features=len(n_headers),
            weights=[0.8,0.2],  #80% of samples are 0, 20% of samples are 1
            shuffle=True,
            random_state=RS)

    # scale the values using a StandardScaler
    scaler = StandardScaler()
    scaler = scaler.fit(x)
    X = scaler.transform(x)
    dump(scaler, 'models/StScaler.joblib') #save the scaler

    #features DataFrame
    features = pd.DataFrame(X, columns = n_headers)

    #-------------------------------------------------------------------------------------------
    #                Pt.2. Training
    #-------------------------------------------------------------------------------------------

    #TODO: here one can play a lot with the model parameters set up

    train_X,test_X, train_y, test_y = train_test_split(features, y, random_state = RS)

    model = RandomForestClassifier(random_state=RS)
    model.fit(train_X, train_y)

    y_pred = model.predict(test_X)
    print("MAE:", mean_absolute_error(test_y, y_pred))
    print("F1 score:", f1_score(test_y, y_pred))
    # plot_norm_conf_matr(test_y, y_pred)

    #save the trained model and the scaler
    dump(model, 'models/RandomForest.joblib')

#-------------------------------------------------------------------------------------------
#                Pt.3. Interference
#-------------------------------------------------------------------------------------------

def inference(row, scaler, model, feat_cols):
    '''Returns the model predictions for particular data samples.

       Args:
          row: particular sample of data
          scaler: scaler for data
          model: NN model
          feat_cols: dataset features the model expects

      # Returns: the models prediction for the row in human readable format
    '''
    df = pd.DataFrame([row], columns = feat_cols)
    X = scaler.transform(df)
    features = pd.DataFrame(X, columns = feat_cols)
    if (model.predict(features)==0):
        return "This is gamma event"
    else: return "This is proton event"

def test_inference():
    #inference input data
    row = a = np.random.rand(len(n_headers))
    model = load( 'models/RandomForest.joblib')
    scaler = load('models/StScaler.joblib')
    print('Result: ', inference(row, scaler, model, n_headers))

if __name__ == '__main__':
    # train_nn()
    test_inference()
