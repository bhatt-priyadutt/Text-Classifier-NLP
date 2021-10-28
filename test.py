import argparse
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
model = load_model("model.h5")
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--testcsv", type=str,help="path",required=True)
args = vars(ap.parse_args())
test = pd.read_csv(args['testcsv'])
test['transcription'] = test['transcription'].str.lower()
puncs = list("?:!.,;") # Replacing punctuations with nothing
for i in puncs:
    test['transcription'] = test['transcription'].str.replace(i,'')
not_to_remove = {'on','off'} # Removing on and off from stopwords
stop = stopwords.words()
stop = [ele for ele in stop if ele not in not_to_remove]
removed_train = []
removed_test = []
for i in range(0,len(test)):
    removed_test.append(" ".join([word for word in str(test['transcription'][i]).split() if word not in stop]))
test['transcription'] = removed_test
tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(test['transcription'])
sequences = tokenizer.texts_to_sequences(test['transcription'])
MAXLEN = 7
X = pad_sequences(sequences, maxlen=MAXLEN)
preds = model.predict(X)
test_df = pd.DataFrame()
test_df['action'] = np.argmax(preds[0],axis=1)
test_df['object'] = np.argmax(preds[1],axis=1)
test_df['location'] = np.argmax(preds[2],axis=1)
print(test_df)