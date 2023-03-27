from flask import Flask, request, render_template
import re
import tensorflow as tf
import pickle


app = Flask(__name__)

#load the model
ml_model = tf.keras.models.load_model('model/LSTM_model')

#prepare the preprocessing tools for the input
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords)
# load the tokenizer from the file
with open('model/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def process_text(input_text):

    #preprocessing
    text = input_text
    text = text.lower() 
    text = REPLACE_BY_SPACE_RE.sub(' ', text) 
    text = BAD_SYMBOLS_RE.sub('', text) 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    text = [text,'']
    text = tokenizer.texts_to_sequences(text)
    text = tf.keras.preprocessing.sequence.pad_sequences(text, maxlen=250)

    #prediction 
    probability = ml_model.predict(text)[0][0]
    result = round(probability)
    result = round(probability*100)

    return result


@app.route('/')
def home():
    result = None
    return render_template('home.html')


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            text = request.form['INPUT-text']
            result = process_text(text)

        except ValueError:
            return "Please check if the words are entered correctly"
        
        return render_template('home.html', result=result)
    

if __name__ == '__main__':
    app.run(host = '0.0.0.0')




# return '''
#         <form method="post">
#             <input type="text" name="text">
#             <input type="submit" value="Submit">
#         </form>
#     '''