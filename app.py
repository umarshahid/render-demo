from flask import Flask, request, render_template
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline

# Load the trained logistic regression model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    logistic_model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

# Initialize the tokenizer and model for Mistral
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

pipe = pipeline(
    "text-generation",
    model=llm_model,
    tokenizer=tokenizer,
    max_length=128
)

hf = HuggingFacePipeline(pipeline=pipe)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = logistic_model.predict(final_features)
    output = 'Placed' if prediction[0] == 1 else 'Not Placed'

    return render_template('index.html', prediction_text='Prediction: {}'.format(output))

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['chat_input']
    response = hf.invoke(user_input)  # Adjust the max_length as needed
    chat_response = response[0]['generated_text']
    return render_template('index.html', chat_response='Response: {}'.format(chat_response))

if __name__ == "__main__":
    app.run(debug=True)
