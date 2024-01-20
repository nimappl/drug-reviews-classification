import csv, re, nltk
from datetime import datetime

train_path = 'drugsComTrain_raw.tsv'
test_path = 'drugsComTest_raw.tsv'
train_size = 20000
test_size = 10000

# fetch and prepare train data
def fetch_dataset(max_size, file_path):
    with open(file_path, 'r', encoding='utf-8', newline='') as dataset:
        tsv_reader = csv.reader(dataset, delimiter='\t')
        data = []

        #Skip the header
        next(tsv_reader, None)

        for _ in range(max_size):
            row = next(tsv_reader)
            # Select only the review and rating
            row = [row[i] for i in [3, 4]]

            # normalize, clean, and tokenize review text
            row[0] = row[0].lower()
            row[0] = row[0].replace('\n', ' ').replace('\r', ' ')
            row[0] = re.sub(r'&(([a-zA-Z]+)|(#\d+));', ' ', row[0])
            row[0] = re.sub(r'[^a-zA-Z\s]', ' ', row[0])
            row[0] = nltk.word_tokenize(row[0])

            # convert ratings to 3 classes
            rating = float(row[1])
            if rating < 4:
                row[1] = 'negative'
            elif rating >= 4 and rating < 7:
                row[1] = 'neutral'
            else:
                row[1] = 'positive'

            data.append(row)
    return data

def time():
    return datetime.now().strftime('%H:%M:%S')

print('size of train set: ', train_size)
print('size of test set: ', test_size)

###################################### SVM ######################################
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# print('------------ support vector machine ------------')
# print(time(), ' retrieving datasets...')
# data_test = fetch_dataset(test_size, test_path)
# data_train = fetch_dataset(train_size, train_path)

# # stem and remove stop words
# print(time(), ' stemming words of reviews and removing stopwords...')
# stop_words = set(stopwords.words('english'))

# for rev in data_train:
#     rev[0] = [word for word in rev[0] if word not in stop_words]
#     rev[0] = ' '.join([nltk.stem.porter.PorterStemmer().stem(w) for w in rev[0]])

# for rev in data_test:
#     rev[0] = [word for word in rev[0] if word not in stop_words]
#     rev[0] = ' '.join([nltk.stem.porter.PorterStemmer().stem(w) for w in rev[0]])

# # Vectorize the text using TF-IDF
# print(time(), ' calculating tf/idf...')
# vectorizer = TfidfVectorizer()
# x_train = vectorizer.fit_transform([text[0] for text in data_train])
# x_test = vectorizer.transform([text[0] for text in data_test])

# # build the model
# model_svm = SVC()
# print(time(), ' training the SVM model using train set...')
# model_svm.fit(x_train, [label[1] for label in data_train]) # train

# print(time(), ' testing the model using test set...')
# y_pred = model_svm.predict(x_test) # test
# y_test = [review[1] for review in data_test]

# # calculate performance metrics
# svm_accuracy = accuracy_score(y_test, y_pred)
# svm_precision = precision_score(y_test, y_pred, average='macro')
# svm_recall = recall_score(y_test, y_pred, average='macro')

# print(f'SVM Accuracy: {svm_accuracy:.4f}')
# print(f'SVM Precision: {svm_precision:.4f}')
# print(f'SVM Recall: {svm_recall:.4f}')

###################################### CNN ######################################
from nltk.stem import WordNetLemmatizer
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, LSTM, MaxPool1D, GlobalMaxPooling1D, Bidirectional, Dense, Dropout, Masking
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.regularizers import l2
from keras.optimizers import Adam

tf.get_logger().setLevel(tf._logging.ERROR)
print('------------ convolutional neural network ------------')
print(time(), 'retrieving datasets...')
data_test = list(); data_train = list()
data_train = fetch_dataset(train_size, train_path)
data_test = fetch_dataset(test_size, test_path)

# lemmatize reviews
print(time(), 'lemmatizing words of reviews...')
for rev in data_train:
    rev[0] = [WordNetLemmatizer().lemmatize(w, 'r') for w in rev[0]]
for rev in data_test:
    rev[0] = [WordNetLemmatizer().lemmatize(w, 'r') for w in rev[0]]

# tokenize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts([r[0] for r in data_train] + [r[0] for r in data_test])
vocab_size = len(tokenizer.word_index.keys())

# retrieve the glov.42B.300d pre-trianed embeddings
print(time(), ' retrieving pre-trained word embeddings...')
embedding_index = {}
with open ('glove.42B.300d.txt', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        if word in tokenizer.word_index.keys():
            coefs = np.array([float(val) for val in values[1:]], dtype='float32')
            embedding_index[word] = coefs

print('size of vocabulary: ', vocab_size)
print('size of embeddings: ', len(embedding_index.keys()))

# make embedding matrix
embedding_matrix = np.zeros((vocab_size + 1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# sequence and pad the data
train_sequences = tokenizer.texts_to_sequences([record[0] for record in data_train])
test_sequences = tokenizer.texts_to_sequences([record[0] for record in data_test])
sequences = tokenizer.texts_to_sequences([record[0] for record in data_train] + [record[0] for record in data_test])
x_train = pad_sequences(train_sequences, maxlen=500)
x_test = pad_sequences(test_sequences, maxlen=500)
x = pad_sequences(sequences, maxlen=500)

# build the model
model_cnn = Sequential()
model_cnn.add(Embedding(vocab_size + 1, 300, input_length=500, weights=[embedding_matrix], trainable=True))
model_cnn.add(Masking(mask_value=0))
model_cnn.add(Bidirectional(LSTM(64, return_sequences=True)))
model_cnn.add(Conv1D(128, 3, padding='same', activation='relu', kernel_regularizer=l2(0.01)))
model_cnn.add(MaxPool1D(2))
model_cnn.add(Conv1D(128, 5, padding='same', activation='relu'))
model_cnn.add(MaxPool1D(2))
model_cnn.add(GlobalMaxPooling1D())
model_cnn.add(Dropout(0.25))
model_cnn.add(Dense(128, activation='relu'))
model_cnn.add(Dropout(0.25))
model_cnn.add(Dense(3, activation='softmax'))

# compile the model
model_cnn.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
print(time(), ' successfully built the model')
print(model_cnn.summary())

# convert labels to numeric format
y_train = np.array([0 if rec[1] == 'negative' else 1 if rec[1] == 'neutral' else 2 for rec in data_train])
y_test = np.array([0 if rec[1] == 'negative' else 1 if rec[1] == 'neutral' else 2 for rec in data_test])

# train the model
# learning rate scheduler
learning_rate_scheduler = LearningRateScheduler(lambda epoch: 0.001 * 0.9**epoch)
# early stopping (to prevent overfitting)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model_cnn.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
              callbacks=[learning_rate_scheduler, early_stopping])

# test the model
print(time(), ' testing...')
y_pred = model_cnn.predict(x_test)
y_pred = np.array([np.argmax(pred) for pred in y_pred])

# calculate performance metrics
cnn_accuracy = accuracy_score(y_test, y_pred)
cnn_precision = precision_score(y_test, y_pred, average='macro')
cnn_recall = recall_score(y_test, y_pred, average='macro')

print(f'CNN Accuracy: {cnn_accuracy:.4f}')
print(f'CNN Precision: {cnn_precision:.4f}')
print(f'CNN Recall: {cnn_recall:.4f}')
