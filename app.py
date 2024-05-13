from flask import Flask, render_template, request, send_file
import pandas as pd
import pickle


app = Flask(__name__)

with open('assets/model.pkl', 'rb') as file:
    model = pickle.load(file)

required_columns = [
            'satisfaction_level', 'last_evaluation', 'number_project',
            'average_montly_hours', 'time_spend_company', 'Work_accident',
            'promotion_last_5years'
        ]

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/single-predict")
def single_predict_page():
    return render_template('single-predict.html')

@app.route('/single-predict', methods=['POST'])
def single_predict():
    required_fields = required_columns + ['department', 'salary']
    if any(request.form[field] == '' for field in required_fields):
        return render_template('single-predict.html', error="All the fields are required!")

    values = [
        float(request.form['satisfaction_level']), #0
        float(request.form['last_evaluation']),
        int(request.form['number_project']),
        int(request.form['average_montly_hours']),
        int(request.form['time_spend_company']),
        int(request.form['Work_accident']),
        int(request.form['promotion_last_5years']), #6
        0,  # cat_IT
        0,  # cat_RandD
        0,  # cat_accounting
        0,  # cat_hr
        0,  # cat_management
        0,  # cat_marketing
        0,  # cat_product_mng
        0,  # cat_sales
        0,  # cat_support
        0,  # cat_technical #16
        0,  # cat_high
        0,  # cat_low
        0   # cat_medium
    ]

    columns = required_columns + [
       'cat_IT', 'cat_RandD', 'cat_accounting',
        'cat_hr', 'cat_management', 'cat_marketing', 'cat_product_mng',
        'cat_sales', 'cat_support', 'cat_technical', 'cat_high',
        'cat_low', 'cat_medium'
    ]

    # Set the appropriate category based on the form input
    values[ columns.index(f'cat_{request.form["department"]}')] = 1
    values[ columns.index(f'cat_{request.form["salary"]}')] = 1

    prediction = model.predict([values])

    return render_template('single-result.html', prediction=prediction[0])


@app.route("/batch-predict")
def batch_predict_page():
    return render_template('batch-predict.html')


@app.route("/batch-predict", methods=['POST'])
def batch_predict():
    file = request.files['file']

    if file.filename == '':
        return render_template('batch-predict.html', error="No selected file!")
    
    if file:
        user_data = pd.read_csv(file)
        user_data_copy = user_data.copy() 

        missing_columns = [col for col in required_columns if col not in user_data.columns]
        if missing_columns:
            return render_template('batch-predict.html', error=f"Required column(s) missing in the uploaded CSV: {missing_columns}. Please refer the sample file!")

        # Load the original data for encoding
        original_data = pd.read_csv('assets/csvs/dataset.csv')
        original_data.drop_duplicates(inplace=True)
        original_data.rename(columns={"sales": "department", "salary": "salary_level"}, inplace=True)

        # Encode categorical columns
        categorical_cols = ["department", "salary_level"]
        categorical_cols_1 = ["department", "salary"]
        encoded_cols_original = pd.get_dummies(original_data[categorical_cols], prefix="cat", dtype=int) 

        # Encode categorical columns in the same way as the training data
        encoded_cols_user = pd.get_dummies(user_data[categorical_cols_1], prefix="cat", dtype=int)

        # Align columns to handle cases where some categorical values are not present in the user data
        encoded_cols_user = encoded_cols_user.reindex(columns=encoded_cols_original.columns, fill_value=0)

        # Concatenate encoded columns with user data
        user_data = pd.concat([user_data, encoded_cols_user], axis=1)

        # Drop original categorical columns and left column
        user_data.drop(["department", "salary", "left"], inplace=True, axis="columns")

        predictions = model.predict(user_data)
        user_data['prediction'] = predictions
        user_data['department'] = user_data_copy['department']

        # Calculate department-wise percentage of employees leaving
        department_leave_counts = user_data.groupby('department')['prediction'].sum()
        department_total_counts = user_data.groupby('department')['prediction'].count()

        # Handle the case where no one is leaving in a department
        department_leave_percentages = (department_leave_counts / department_total_counts).fillna(0) * 100
        return render_template('batch-results.html', result_table=user_data, department_leave_percentages=department_leave_percentages)


@app.route("/download-sample-csv")
def download_sample():
    return send_file('assets/csvs/sample.csv', as_attachment=True, download_name='sample.csv')

if __name__ ==  "__main__":
    app.run()