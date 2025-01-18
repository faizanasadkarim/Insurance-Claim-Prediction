from flask import Flask, render_template, request
import pickle
import pandas as pd

model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
app = Flask(__name__)

def convert_to_correct_type(form_input):
    """Convert input fields to their correct types (e.g., numeric)."""
    for key, value in form_input.items():
        
        try:
            # Try to convert to float
            form_input[key] = float(value)
        except ValueError:
            # Leave it as a string if neither conversion worked
            pass
    return form_input


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        form_input = request.form.to_dict()
        # Convert form inputs to their correct types
        form_input = convert_to_correct_type(form_input)
        # Create a DataFrame with one row
        df_input = pd.DataFrame([form_input])
        x=encoder.transform(df_input)
        pred=model.predict(x)
        return str(pred[0])

if __name__ == "__main__":
    app.run(debug=True)
