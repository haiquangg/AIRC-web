import torch
import flask
from flask import Flask, request, render_template
import json
from transformers import BartTokenizer, BartForConditionalGeneration

app = Flask(__name__)
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
BART_PATH = "sshleifer/distilbart-cnn-6-6"
bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH, output_past=True)

# t5_model = T5ForConditionalGeneration.from_pretrained(T5_PATH)
# t5_tokenizer = T5Tokenizer.from_pretrained(T5_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def bart_summarize(input_text, num_beams=4, num_words=50):
    input_text = str(input_text)
    input_text = ' '.join(input_text.split())
    input_tokenized = bart_tokenizer.encode(input_text, return_tensors='pt').to(device)
    summary_ids = bart_model.generate(input_tokenized,
                                      num_beams=int(num_beams),
                                      no_repeat_ngram_size=3,
                                      length_penalty=2.0,
                                      min_length=30,
                                      max_length=int(num_words),
                                      early_stopping=True)
    output = [bart_tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids]
    return output[0]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sentence = request.json['input_text']
        num_words = request.json['num_words']
        num_beams = request.json['num_beams']
        # model = request.json['model']
        if sentence != '':
            # if model.lower() == 'customer':
            output = bart_summarize(sentence, num_beams, num_words)
            response = {}
            response['response'] = {
                'summary': str(output)
            }
            return flask.jsonify(response)
        else:
            res = dict({'message': 'Empty input'})
            return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')
    except Exception as ex:
        res = dict({'message': str(ex)})
        print(res)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    bart_model.to(device)
    bart_model.eval()
    app.run(host='0.0.0.0', debug=True, port=8000, use_reloader=False)
