from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask Web Application
application = Flask(__name__)
app = application

# Route 1: The Home Page (What the user sees when they load the website)
@app.route('/')
def index():
    return render_template('index.html')

# Route 2: The Prediction Engine (What happens when they click "Predict")
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # 1. Catch the data from the HTML form
            data = CustomData(
                credit_score=int(request.form.get('credit_score')),
                country=request.form.get('country'),
                gender=request.form.get('gender'),
                age=int(request.form.get('age')),
                tenure=int(request.form.get('tenure')),
                balance=float(request.form.get('balance')),
                products_number=int(request.form.get('products_number')),
                credit_card=int(request.form.get('credit_card')),
                active_member=int(request.form.get('active_member')),
                estimated_salary=float(request.form.get('estimated_salary'))
            )
            
            # 2. Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            
            # 3. Send to the Predict Pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)
            
            # 4. Translate the 0 or 1 into English
            if results[0] == 1:
                final_result = "⚠️ HIGH RISK: This customer is likely to CHURN."
            else:
                final_result = "✅ SAFE: This customer is likely to STAY."
                
            return render_template('index.html', results=final_result)
            
        except Exception as e:
            return f"An error occurred: {e}"

if __name__=="__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)