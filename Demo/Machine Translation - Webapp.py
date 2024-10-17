from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from werkzeug.utils import secure_filename
import pysrt
import os
import json
import time
import requests
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel
import pickle
import torch
import numpy as np
from flask_cors import CORS  # Added for handling CORS

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

metis_api_key = "tpsg-YkxgxrCSgCmZSxENz1Mcrsy2Ph3lPot"

model_provider = "openai_chat_completion"
model_name = "gpt-4o"

class MetisLLM(LLM):
    def _call(self, prompt, stop=None, run_manager=None, **kwargs):
        return request_to_metis(prompt, metis_api_key, model_provider, model_name)
    
    @property
    def _identifying_params(self):
        return {"model": model_name}
    
    @property
    def _llm_type(self):
        return "metis_llm"

def request_to_metis(prompt, api_key, model_provider, model_name):
    metis_url = 'https://api.metisai.ir/api/v1/chat/{provider}/completions'

    headers = {
        "x-api-key": api_key,
        "content-Type": "application/json"
    }

    data = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "you are a helpful assistant that translates English subtitle of a movie to Persian"
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    url = metis_url.replace('{provider}', model_provider)

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Failed to get response from Metis API: {response.text}")

def metis_translate(sentence):
    model = MetisLLM()
    template = """use informal language to translate the following English text to Persian:{input_text}."""
    Prompt_template = PromptTemplate(template=template, input_variables=["input_text"])
    messages = Prompt_template.format(input_text=sentence)
    output_sentence = model.invoke(messages)
    return output_sentence

# Specify the path where the trained model is saved
bart_model_path = r"C:\Users\SnappFood\Downloads\trained_mbart_model"

# Load the tokenizer and model
bart_tokenizer = MBart50TokenizerFast.from_pretrained(bart_model_path)
bart_model = MBartForConditionalGeneration.from_pretrained(bart_model_path)

# Set the source and target languages
source_lang = "en_XX"
target_lang = "fa_IR"
bart_tokenizer.src_lang = source_lang
bart_tokenizer.tgt_lang = target_lang

# Define the translation function
def bart_translate(sentence):
    # Encode the input sentence
    encoded_input = bart_tokenizer(sentence, return_tensors="pt", max_length=128, truncation=True)

    # Generate translation using the model
    generated_tokens = bart_model.generate(
        **encoded_input,
        max_length=128,
        num_beams=4,
        early_stopping=True,
        forced_bos_token_id=bart_tokenizer.lang_code_to_id[target_lang]
    )

    # Decode the generated tokens into the target language
    translated = bart_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return translated

mt5_model_path = r"C:\Users\SnappFood\Downloads\MT5 Model"

mt5_model_name = "persiannlp/mt5-small-parsinlu-translation_en_fa"
mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_model_name)
mt5_model = T5ForConditionalGeneration.from_pretrained(mt5_model_path)

def mt5_translate(text):
    mt5_model.eval()
    input_ids = mt5_tokenizer(text, return_tensors="pt").input_ids
    outputs = mt5_model.generate(input_ids, max_new_tokens=500)
    decoded_text = mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_text

### RNN

rnn_model_path = r"C:\Users\SnappFood\Desktop\Rahnema college\model_rnn.h5"

# Load the model
rnn_model = load_model(rnn_model_path)

# Load BERT tokenizer and model
rnn_bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
rnn_bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

rnn_tokenizer_path = r"C:\Users\SnappFood\Desktop\Rahnema college\rnn-tokenizer.pickle"
with open(rnn_tokenizer_path, 'rb') as handle:
    rnn_tokenizer = pickle.load(handle)

def get_bert_embeddings(sentences, max_length=20):
    embeddings = []
    for sentence in sentences:
        inputs = rnn_bert_tokenizer(sentence, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        with torch.no_grad():
            outputs = rnn_bert_model(**inputs)
            embeddings.append(outputs.last_hidden_state.squeeze(0).numpy())
    return np.array(embeddings)

def rnn_translate(sentence):
    try:
        max_length = 20
        embeddings = get_bert_embeddings([sentence], max_length)
        embeddings = embeddings.astype('float32')

        predictions = rnn_model.predict(embeddings)
        predicted_sentence = []

        for i in range(predictions.shape[1]):
            predicted_token = np.argmax(predictions[0, i])
            if predicted_token != 0:
                predicted_word = rnn_tokenizer.index_word.get(predicted_token, '')
                if predicted_word:
                    predicted_sentence.append(predicted_word)
        
        translated_text = ' '.join(predicted_sentence)
        return translated_text
    
    except Exception as e:
        return f"Error during translation: {str(e)}"

### LSTM

lstm_model_path = r"C:\Users\SnappFood\Desktop\Rahnema college\model_lstm.h5"
lstm_model = load_model(lstm_model_path)

lstm_tokenizer_path = r"C:\Users\SnappFood\Desktop\Rahnema college\lstm tokenizer .pickle"
with open(lstm_tokenizer_path, 'rb') as handle:
    lstm_tokenizer = pickle.load(handle)

lstm_bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
lstm_bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

def lstm_translate(sentence):
    try:
        max_length = 20
        embeddings = get_bert_embeddings([sentence], max_length)
        embeddings = embeddings.astype('float32')
        predictions = lstm_model.predict(embeddings)
        predicted_sentence = []
        for i in range(predictions.shape[1]):
            predicted_token = np.argmax(predictions[0, i])
            if predicted_token != 0:
                predicted_word = lstm_tokenizer.index_word.get(predicted_token, '')
                if predicted_word:
                    predicted_sentence.append(predicted_word)
        translated_text = ' '.join(predicted_sentence)
        return translated_text
    except Exception as e:
        return f"Error during translation: {str(e)}"

# Pre-load all models when the server starts
models = {
    "llm": metis_translate,
    "lstm": lstm_translate,
    "rnn": rnn_translate,
    "bart": bart_translate,
    "mt5": mt5_translate
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    return jsonify({"message": "File uploaded successfully"}), 200

@app.route('/translate_line/<model_type>', methods=['GET'])
def translate_line(model_type):
    translator = models.get(model_type)
    if not translator:
        return jsonify({"error": f"Model type {model_type} is not supported"}), 400

    uploaded_files = os.listdir(app.config["UPLOAD_FOLDER"])
    if not uploaded_files:
        return jsonify({"error": "No uploaded file found"}), 400
    
    latest_file = max(
        [os.path.join(app.config["UPLOAD_FOLDER"], f) for f in uploaded_files],
        key=os.path.getctime
    )

    subs = pysrt.open(latest_file)
    total_subs = len(subs)

    @stream_with_context
    def generate():
        for idx, sub in enumerate(subs):
            print(f"Translating line {sub.index}/{total_subs}")
            english_text = sub.text
            translated_text = translator(english_text)
            translation_data = {
                "index": sub.index,
                "original_text": english_text,
                "translated_text": translated_text
            }

            yield f"data: {json.dumps(translation_data)}\n\n"
            time.sleep(0.1)

    return Response(generate(), content_type='text/event-stream')

@app.route('/translate_text/<model_type>', methods=['POST'])
def translate_text(model_type):
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    translator = models.get(model_type)
    if not translator:
        return jsonify({"error": f"Model type {model_type} is not supported"}), 400

    translated_text = translator(text)
    return jsonify({"translated_text": translated_text}), 200

if __name__ == '__main__':
    app.run(debug=True)