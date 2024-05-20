import json
import pickle
import nltk
import random
import numpy as np

from nltk.stem import WordNetLemmatizer
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Inicializamos nossas listas
words = []
documents = []
intents = json.loads(open('intents.json').read())
classes = [i['tag'] for i in intents['intents']]
ignore_words = ["!", "@", "#", "$", "%", "*", "?"]

# Processamos as intenções
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Inicializamos o treinamento
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Embaralhamos e verificamos a consistência dos dados
random.shuffle(training)

# Verifique a consistência dos dados
for i in range(len(training)):
    if len(training[i][0]) != len(words) or len(training[i][1]) != len(classes):
        print(f"Inconsistent data at index {i}: {training[i]}")

# Transformamos em numpy array
training = np.array(training, dtype=object)  # Use dtype=object to handle nested lists of different lengths

# Criamos listas de treino
x = np.array([i[0] for i in training])
y = np.array([i[1] for i in training])

# Verifique as dimensões
print(f"x shape: {x.shape}, y shape: {y.shape}")

# Construímos o modelo
model = Sequential()
model.add(Dense(128, input_shape=(len(x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y[0]), activation='softmax'))

# Compilamos o modelo
sgd = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Ajustamos o modelo e salvamos
m = model.fit(np.array(x), np.array(y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5', m)

print("fim")
