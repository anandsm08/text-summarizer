from flask import Flask, render_template, request
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch

app = Flask(__name__)


model_name = "google/pegasus-xsum"
tokenizer = PegasusTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize-text',methods=["POST"])
def summarize():
    if request.method == "POST":
        text_input = request.form["inputtext_"]
        input_text = "Summarized text : "+ text_input

        text_token=tokenizer.encode(input_text,return_tensors='pt',max_length=1024).to(device)
        summary_ = model.generate(text_token, min_length=30, max_length=512)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        print(summary)
    return render_template('output.html', data ={"summary":summary})

if __name__ == '__main__':
    app.run()
