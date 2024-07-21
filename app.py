import os
from flask import Flask, request, render_template
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEndpoint


# Load the trained logistic regression model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    logistic_model = pickle.load(file)

# Initialize the Flask app
app = Flask(__name__)

api_key = os.getenv("HF_API_KEY")
# api_key = "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"
# print(f"API Key: {api_key}")
# repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# llm_model = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=api_key, )

# Initialize the tokenizer and model for Mistral
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_auth_token=api_key)
# tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
llm_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")

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
     # Debugging
    print("Response:", response)
    print("Type of Response:", type(response))
    
    # Process the response based on its type
    if isinstance(response, str):
        import json
        response = json.loads(response)
    
    # Adjust based on the structure of the response
    if isinstance(response, list):
        chat_response = response[0].get('generated_text', 'No generated text found')
    elif isinstance(response, dict):
        chat_response = response.get('generated_text', 'No generated text found')
    else:
        chat_response = 'Unexpected response format'
    
    return render_template('index.html', chat_response='Response: {}'.format(chat_response))

if __name__ == "__main__":
    app.run(debug=True)
