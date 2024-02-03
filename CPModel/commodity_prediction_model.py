#importing the required python libraries
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
from datetime import datetime


#generates features
def feature_generator(dF):
    dF=dF.copy()
    dF['year']=dF.index.year#year
    dF['month']=dF.index.month #we get month from our datetime index
    
    return dF

def infer_model(county,start_date,end_date,pstr_date,pend_date):
    dataframe=pd.read_csv('./Data/data.csv',usecols=['periodid',county]).dropna()
    dataframe['periodid']=pd.to_datetime(dataframe['periodid'])
    dset_indexed=dataframe.set_index(['periodid'])
    
    #train and test dataset
    train=dset_indexed.loc[dset_indexed.index<'01-01-2021']
    test=dset_indexed.loc[dset_indexed.index>='01-01-2021'] 
    
    #generate feature from indexed data
    df=feature_generator(dset_indexed)
    
    #import necessary libraries then make pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    
    #run train and test through features function to create features we are going to use
    train_set=feature_generator(train)
    test_set=feature_generator(test)
    
    FEATURES=['year','month']
    TARGET=county
    
    X_train=train_set[FEATURES]
    y_train=train_set[TARGET]

    X_test=test_set[FEATURES]
    y_test=test_set[TARGET]
    
    #model pipeline
    model_pipeline=Pipeline([('scaler',StandardScaler()),('gbr',GradientBoostingRegressor(n_estimators=2500,loss='absolute_error',learning_rate=0.001))])
    
    #we fit the ML pipeline
    model_pipeline.fit(X_train,y_train)
    
    #add the predicted values to the dataframe
    test['prediction']=model_pipeline.predict(X_test)
    df=df.merge(test[['prediction']], how='left',left_index=True,right_index=True)
    
    #project prediction
    start_date=pstr_date.strftime('%Y-%m')
    end_date=pend_date.strftime('%Y-%m')

    dates_array=[]
    period_array=[]

    dates=pd.date_range(start_date,end_date)
    for date in dates:
        dates_array.append([date.year,date.month])
        period_array.append(f"{date.year}-{date.month}")
    transf_dates_array=np.unique(dates_array,axis=0)
    period_array=np.unique(period_array,axis=0)
    
    #the prediction
    date_input=pd.DataFrame(data=transf_dates_array,columns=['year','month'])
    
    #predict using model in pipleine
    y_hat=model_pipeline.predict(date_input)

    # Make predictions using the model
    predictions = y_hat
    #add to the data_frame new values
    dataframe[county] = pd.concat([pd.Series(predictions),dataframe[f"{county}"]],ignore_index=True)
    dataframe['periodid'] = pd.concat([pd.Series(period_array),dataframe['periodid']],ignore_index=True)
    dataframe['periodid']=pd.to_datetime(dataframe['periodid'])
    #arrange in descending order
    #dataframe.sort_values(by='periodid',inplace=True,ascending=False)
    #arrange in ascending by default
    dataframe.sort_values(by='periodid',inplace=True)
    dataframe.reset_index(drop=True, inplace=True)
    dset_indexed=dataframe.set_index(['periodid'])

     #generate feature from indexed data
    df=np.array(([dset_indexed.index.astype(str)],[dset_indexed[county]]))
    #check what you are sending
    #print(dataframe.head(10))

    return predictions,df


def get_dataframe(county):
    dataframe=pd.read_csv('./Data/data.csv',usecols=['periodid',county]).dropna()
    dataframe['periodid']=pd.to_datetime(dataframe['periodid'])
    dset_indexed=dataframe.set_index(['periodid'])
    
    #generate feature from indexed data
    df=np.array(([dset_indexed.index.astype(str)],[dset_indexed[county]]))

    return df



    
    
    