from datetime import datetime
from flask import Flask, jsonify,render_template,flash,redirect, template_rendered,url_for,session,logging,request
from flask_cors import CORS
import os

#prediction model
from CPModel.commodity_prediction_model import infer_model
from CPModel.commodity_prediction_model import get_dataframe
#from CPModel.Nakuru_model import timeseries_predictor

app=Flask(__name__)
CORS(app) #add this for fetch API

def Reverse(lst):
   new_lst = lst[::-1]
   return new_lst

@app.route('/api/v1.1/analysis',methods=['POST'])
def analysis():
    try:
        if request.method == 'POST':
            request_body = request.get_json()
            # Extract relevant information from the request data
            county_name = request_body.get('county_name', '')
            fstart_date = request_body.get('fstart_date', '')
            fend_date = request_body.get('fend_date', '')
            fstr = request_body.get('fstr', '')
            fend = request_body.get('fend', '')
            
            fstart_date=datetime.strptime(fstart_date,'%Y-%m-%d')
            fend_date=datetime.strptime(fend_date,'%Y-%m-%d')
            pred_strdate=datetime.strptime(fstr,'%Y-%m-%d')
            pred_enddate=datetime.strptime(fend,'%Y-%m-%d')
            pred,dataFrame=infer_model(county_name,fstart_date,fend_date,pred_strdate,pred_enddate)
            pred=pred.tolist()

            #get the new series dataframe
            series = dataFrame.tolist()

            #reverse the dates and the values
            """series[0][0] = series[0][0][::-1]
            series[1][0] = series[1][0][::-1]"""
            
            return jsonify({
                "county":county_name,
                "startDate":fstart_date,
                "endDate":fend_date,
                "predStrdate":pred_strdate,
                "predEnddate":pred_enddate,
                "prediction":pred,
                "response":series
                }),200
        else:
            return jsonify({
                "status":"failed",
                "response":"bad request sent by client"
            }),400
    except Exception as e:
        return jsonify({
            "status":"unknown error",
            "response":{"Error":f"{e}"}
        }),400



@app.route('/api/v1.1/timeseries',methods=['POST'])
def getSeriesData():
    try:
        if request.method == 'POST':
            request_body = request.get_json()
            series = get_dataframe(request_body.get('county_name','')).tolist()

            #reverse the dates and the values
            series[0][0] = series[0][0][::-1]
            series[1][0] = series[1][0][::-1]

            return jsonify(
                {
                    "status":"success",
                    "response": series
                }
            ),200
    except Exception as e:
        return jsonify({
            "status":"failed",
            "response":{"Error":f"{e}"}
        }),400



if __name__=='__main__':
    app.secret_key='key123'
    app.run(debug=True,host='0.0.0.0',port=8080)