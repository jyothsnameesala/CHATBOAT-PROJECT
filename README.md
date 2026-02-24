Detailed Code Explanation with Key Snippets
Train_chatbot.py
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import SGD
import random
# Load intents file
data_file = open('intents.json').read()
intents = json.loads(data_file)
words = []
classes = []
documents = []
ignore_words = ['?', '!']
# Process intents
for intent in intents['intents']:
 for pattern in intent["patterns"]:
 # Tokenize each word
 w = nltk.word_tokenize(pattern)
 words.extend(w)
 # Add documents in the corpus
 documents.append((w, intent['tag']))
 # Add to our classes list
 if intent['tag'] not in classes:
 classes.append(intent['tag'])
# Lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# Sort classes
classes = sorted(list(set(classes)))
# Print information
13 | P a g e
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)
# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
# Create our training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
 bag = []
 pattern_words = doc[0]
 pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
 for w in words:
 bag.append(1) if w in pattern_words else bag.append(0)
 output_row = list(output_empty)
 output_row[classes.index(doc[1])] = 1
 training.append([bag, output_row])
# Shuffle our features and turn into np.array
random.shuffle(training)
14 | P a g e
# Print shapes of elements
for item in training:
 print(len(item), len(item[0]), len(item[1]))
# Ensure consistent shapes
training = np.array(training, dtype=object)
# Create train and test lists. X - patterns, Y - intents
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])
print("Training data created")
# Create model
model = Sequential()
model.add(Input(shape=(len(train_x[0]),)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax
# Compile model
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
15 | P a g e
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# Fit and save the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")
Chatgui.py
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
def clean_up_sentence(sentence):
 # tokenize the pattern - split words into array
 sentence_words = nltk.word_tokenize(sentence)
 # stem each word - create short form for word
 sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
 return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
 # tokenize the pattern
 sentence_words = clean_up_sentence(sentence)
 # bag of words - matrix of N words, vocabulary matrix
 bag = [0]*len(words)
 for s in sentence_words:
 for i,w in enumerate(words):
 if w == s: 
# assign 1 if current word is in the vocabulary position
 bag[i] = 1
 if show_details:
 print ("found in bag: %s" % w)
 return(np.array(bag))
def predict_class(sentence, model):
 # filter out predictions below a threshold
 p = bow(sentence, words,show_details=False)
 res = model.predict(np.array([p]))[0]
 ERROR_THRESHOLD = 0.25
 results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
 # sort by strength of probability
 results.sort(key=lambda x: x[1], reverse=True)
 return_list = []
 for r in results:
 return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
 return return_list
def getResponse(ints, intents_json):
 tag = ints[0]['intent']
 list_of_intents = intents_json['intents']
 for i in list_of_intents:
 if(i['tag']== tag):
 result = random.choice(i['responses'])
 break
 return result
def chatbot_response(msg):
 ints = predict_class(msg, model)
 res = getResponse(ints, intents)
 return res
#Creating GUI with tkinter
import tkinter
from tkinter import *
def send():
 msg = EntryBox.get("1.0",'end-1c').strip()
 EntryBox.delete("0.0",END)
 if msg != '':
 ChatLog.config(state=NORMAL)
 ChatLog.insert(END, "You: " + msg + '\n\n')
 ChatLog.config(foreground="#442265", font=("Verdana", 12 ))

 res = chatbot_response(msg)
 ChatLog.insert(END, "Bot: " + res + '\n\n')
 ChatLog.config(state=DISABLED)
 ChatLog.yview(END)
base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
#Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatLog.config(state=DISABLED)
#Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
#Create Button to send message
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
 bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
 command= send )
#Create the box to enter message
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()
Dataset
{
 "intents":
 [
 {"tag": "greeting",
 "patterns": ["Hi there", "How are you", "Is anyone there?","Hey","Hola", "Hello",
"Good day"],
 "responses": ["Hello, thanks for asking", "Good to see you again", "Hi there, how can I
help?"],
 "context": [""]
 },
 {"tag": "goodbye",
 "patterns": ["Bye", "See you later", "Goodbye", "Nice chatting to you, bye", "Till next
time"],
 "responses": ["See you!", "Have a nice day", "Bye! Come back again soon."],
 "context": [""]
 },
 {"tag": "thanks",
 "patterns": ["Thanks", "Thank you", "That's helpful", "Awesome, thanks", "Thanks for
helping me"],
 "responses": ["Happy to help!", "Any time!", "My pleasure"],
 "context": [""]
 },
 {"tag": "noanswer",
 "patterns": [],
 "responses": ["Sorry, can't understand you", "Please give me more info", "Not sure I
understand"],
 "context": [""]
 },
 {"tag": "options",
 "patterns": ["How you could help me?", "What you can do?", "What help you
provide?", "How you can be helpful?", "What support is offered"],
 "responses": ["I can guide you through Adverse drug reaction list, Blood pressure
tracking, Hospitals and Pharmacies", "Offering support for Adverse drug reaction, Blood
pressure, Hospitals and Pharmacies"],
 "context": [""]
 },
 {
 "tag": "age",
 "patterns": ["How old are you?", "What's your age?", "Can you tell me your age?"],
 "responses": ["I'm a timeless entity.", "I don't have an age.", "Age is just a number!"],
 "context": [""]
 },
 {

 "tag": "weather",
 "patterns": ["What's the weather like?", "Tell me the current weather.", "How's the
weather today?"],
 "responses": ["I can't check the weather right now, but you can check a weather website
for the latest updates.", "Please check your local weather service for the latest information."],
 "context": [""]
 },
 {
 "tag": "news",
 "patterns": ["What's the latest news?", "Tell me the news.", "Any breaking news?"],
 "responses": ["I can't provide real-time news updates, but you can check a news website 
 for the latest information.", "Please refer to a trusted news source for the latest updates."],
 "con
 },
  {
  "tag": "joke",
 "patterns": ["Tell me a joke", "Do you know any jokes?", "Make me laugh"],
 "responses": ["Why don't scientists trust atoms? Because they make up everything!",
"Why did the scarecrow win an award? Because he was outstanding in his field!"],
 "context": [""]
 }
 }
