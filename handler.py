import os
import pickle
import pandas as pd
from flask              import Flask, request ,Response
from rossmann.Rossmann  import Rossmann

# load model
model = pickle.load( open('model/model_rossmann.pk1', 'rb') )

# inicialize API
app = Flask( __name__ )

@app.route('/rossmann/predict', methods = ['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json:
        
        if isinstance(test_json, dict): # Unique Example
            test_raw = pd.DataFrame( test_json, index=[0] )
        
        else: # Multiple Example
            test_raw = pd.DataFrame( test_json, columns = test_json[0].keys())
    
        # Instanciar a Rossmann Class
        pipeline = Rossmann()
        
        # Data Cleanning
        df1 = pipeline.data_clenning(test_raw)
        # Feature Engeneering
        df2 = pipeline.feature_engineering(df1)
        # Data Preparation
        df3 = pipeline.data_preparation(df2)
        # Prediction
        df_responde = pipeline.get_prediction(model, test_raw, df3)

        return df_responde
    
    else:
        return Response({}, status=200, minetype='application/json')

if __name__ == '__main__':
    port = int( os.environ.get( 'PORT', 5000 ) )
    app.run( '0.0.0.0', port=port)