from flask import Flask, request, jsonify, render_template
import random
import warnings
import nltk
from newspaper import Article
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

warnings.filterwarnings('ignore')
nltk.download('punkt', quiet=True)

# Download and parse the article
article_url = 'https://www.mayoclinic.org/diseases-conditions/chronic-kidney-disease/symptoms-causes/syc-20354521'
article = Article(article_url)
article.download()
article.parse()
article.nlp()
corpus = article.text

# Tokenize the corpus into sentences
sentence_list = nltk.sent_tokenize(corpus)

# Greeting response function
def greeting_response(text):
    text = text.lower()
    bot_greetings = ['hi', 'hello', 'hey', 'halo', 'greetings', 'howdy', 'hi there', 'hello there']
    user_greetings = ['hi', 'hello', 'greetings', 'whatsapp', 'hey', 'halo', 'howdy', 'good morning', 'good evening']

    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)
    return None

# Specific responses dictionary
specific_responses = {
    'what is chronic kidney disease': 'Chronic kidney disease (CKD) means your kidneys are damaged and can’t filter blood the way they should.',
    'symptoms of chronic kidney disease': 'Symptoms of CKD include nausea, vomiting, loss of appetite, fatigue, and changes in urination.',
    'causes of chronic kidney disease': 'Common causes of CKD include diabetes, high blood pressure, and other conditions.',
    'treatment for chronic kidney disease': 'Treatments for CKD include lifestyle changes, medications, and in severe cases, dialysis or kidney transplant.',
    'prevent chronic kidney disease': 'Preventive measures for CKD include managing diabetes and high blood pressure, maintaining a healthy weight, and avoiding smoking.',
    'diet for chronic kidney disease': 'A kidney-friendly diet includes limiting sodium, potassium, and phosphorus, and eating high-quality protein.',

    'who are you': 'I am Visab Bot, your healthcare assistant.',
    'thank you': 'You are welcome!',
    'thanks': 'You are welcome!',
    'bye': 'Goodbye! Take care!',
    'what can you do': 'I can provide information on chronic kidney disease based on the data I have.',
    'what is your name': 'My name is Visab Bot.',
    'how are you': 'I am just a bot, but I am here to help you!'

}

def get_specific_response(user_input):
    user_input = user_input.lower()
    for prompt, response in specific_responses.items():
        if prompt in user_input:
            return response
    return None

# Function to sort the index of similarity scores
def index_sort(list_var):
    length = len(list_var)
    list_index = list(range(0, length))

    for i in range(length):
        for j in range(length):
            if list_var[list_index[i]] > list_var[list_index[j]]:
                temp = list_index[i]
                list_index[i] = list_index[j]
                list_index[j] = temp
    return list_index

# Generate bot response
def bot_response(user_input):
    user_input = user_input.lower()
    
    specific_response = get_specific_response(user_input)
    if specific_response:
        return specific_response
    
    sentence_list.append(user_input)
    cm = CountVectorizer().fit_transform(sentence_list)
    similarity_scores = cosine_similarity(cm[-1], cm)
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list)
    index = index[1:]
    
    response_flag = 0
    max_words = 80  # Maximum number of words in the response
    bot_response = ''
    word_count = 0
    
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0:
            # Split the sentence into words
            sentence = sentence_list[index[i]]
            words = sentence.split()
            
            if word_count + len(words) > max_words:
                # If adding this sentence exceeds the limit, truncate it
                remaining_words = max_words - word_count
                bot_response += ' ' + ' '.join(words[:remaining_words])
                break
            else:
                bot_response += ' ' + sentence
                word_count += len(words)
                response_flag = 1
                
        if response_flag > 2:
            break

    if response_flag == 0:
        bot_response += " I apologize, I don't understand. Can you please ask something else?"
    sentence_list.remove(user_input)

    return bot_response


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["GET"])
def get_bot_response():
    user_text = request.args.get('msg')
    if greeting_response(user_text) is not None:
        return str(greeting_response(user_text))
    else:
        return str(bot_response(user_text))

@app.route("/predict", methods=["POST"])
def predict():
    user_input = request.json.get('message')
    if greeting_response(user_input) is not None:
        response = greeting_response(user_input)
    else:
        response = bot_response(user_input)
    return jsonify({"answer": response})

if __name__ == "__main__":
    app.run(debug=True)
