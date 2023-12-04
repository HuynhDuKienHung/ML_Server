from flask import Flask , render_template, request,jsonify
import json
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import model

app = Flask(__name__)
#declared an empty variable for reassignment
response = ''

@app.route("/")
def hello():
    return "hello"

@app.route("/", methods = ['POST'])
def submit():
    if request.method == "POST":
        try:
            request_data = request.get_json()
#--------------------------------------------------------------------------------------------
            Arg1_gender = request_data['arg_1']
            Arg2_age = request_data['arg_2']
            Arg3_hypertension = request_data['arg_3']
            Arg4_heart_disease = request_data['arg_4']
            Arg5_ever_married = request_data['arg_5']
            Arg6_work_type = request_data['arg_6']
            Arg7_Residence_type = request_data['arg_7']
            Arg8_avg_glucose_level = request_data['arg_8']
            Arg9_bmi = request_data['arg_9']
            Arg10_smoking_status = request_data['arg_10']
        
#------------------xu ly du lieu----------------------------------------------------------------
    #        Arg1_gender = ast.literal_eval(Arg1_gender);
    #        Arg5_ever_married = ast.literal_eval(Arg5_ever_married);
    #        Arg6_work_type = ast.literal_eval(Arg6_work_type)
    #        Arg7_Residence_type = ast.literal_eval(Arg7_Residence_type)
    #        Arg10_smoking_status = ast.literal_eval(Arg10_smoking_status)
            
            Arg3_hypertension = int(Arg3_hypertension)
            Arg4_heart_disease = int(Arg4_heart_disease)

            Arg2_age = float(Arg2_age)
            Arg8_avg_glucose_level = float(Arg8_avg_glucose_level)
            Arg9_bmi = float(Arg9_bmi)

#------------------------------------------------------------------------------------------------
            new_data = pd.DataFrame(
                [[Arg1_gender, Arg2_age,Arg3_hypertension,Arg4_heart_disease,Arg5_ever_married,Arg6_work_type,Arg7_Residence_type,Arg8_avg_glucose_level,Arg9_bmi,Arg10_smoking_status, 0]],
                columns=['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                         'avg_glucose_level', 'bmi', 'smoking_status', 'stroke'])
            print(new_data)


            dataframe = pd.read_csv('Strokesdataset_Harvard.csv')
            dataframe = dataframe.drop('id',axis = 1)
            dataframe = dataframe.dropna()            
            Cols = ['gender','ever_married','work_type','Residence_type','smoking_status']
            Columns = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status','stroke']

            DataPrep = pd.concat([new_data, dataframe], ignore_index=True)
 #           print(DataPrep)



            DataPrep[Cols] = DataPrep[Cols].astype('category')
            for Columns in Cols:
                DataPrep[Columns] = DataPrep[Columns].cat.codes

#--------------Data Normalization-----------------------------------------------------------------
            scaler = MinMaxScaler()
            DataPrep = pd.DataFrame(scaler.fit_transform(DataPrep), columns=DataPrep.columns)
            input = DataPrep.head(1);

            input = input.drop('stroke', axis=1)
            print(input)
            print()
            Result = model.rfc.predict(input)
            print(Result)
            return jsonify(Result.tolist())
        except json.JSONDecodeError as e:
            return jsonify({'error': f'Lỗi khi giải mã JSON: {e}'})
        except Exception as e:
            return jsonify({'error': f'Có lỗi xảy ra: {e}'})
    else:
        return "Phương thức yêu cầu không hợp lệ"
#--------------------------------------------------------


if __name__ == "__main__":
    app.run(debug= True)