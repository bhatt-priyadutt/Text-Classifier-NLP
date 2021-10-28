import argparse
import yaml
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation, Input
from keras.layers.embeddings import Embedding
import datetime,os
import tensorflow as tf
from sklearn.metrics import f1_score
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--config", type=str,help="path",default="config.yaml")
args = vars(ap.parse_args())
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = len(tf.config.list_physical_devices('GPU'))
if gpus > 1:
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
         tf.config.LogicalDeviceConfiguration(memory_limit=1024)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
with open(args['config']) as file:
  yaml_data= yaml.safe_load(file)
nltk.download('stopwords')
df = pd.read_csv('train_data.csv')
test = pd.read_csv('valid_data.csv')
df['transcription'] = df['transcription'].str.lower() #Lower case
test['transcription'] = test['transcription'].str.lower()
puncs = list("?:!.,;") # Replacing punctuations with nothing
for i in puncs:
    df['transcription'] = df['transcription'].str.replace(i,'')
    test['transcription'] = test['transcription'].str.replace(i,'')
not_to_remove = {'on','off'} # Removing on and off from stopwords
stop = stopwords.words()
stop = [ele for ele in stop if ele not in not_to_remove]
removed_train = []
removed_test = []
for i in range(0,len(df)):
    removed_train.append(" ".join([word for word in str(df['transcription'][i]).split() if word not in stop]))
for i in range(0,len(test)):
    removed_test.append(" ".join([word for word in str(test['transcription'][i]).split() if word not in stop]))
df['transcription'] = removed_train
test['transcription'] = removed_test
df1 = df.copy()
test1 = test.copy()
labelencoder_a=LabelEncoder()
labelencoder_o=LabelEncoder()
labelencoder_l=LabelEncoder()
labelencoder_a_test=LabelEncoder()
labelencoder_o_test=LabelEncoder()
labelencoder_l_test=LabelEncoder()
y_test_action = labelencoder_a_test.fit_transform(test1['action'].values)
y_test_object = labelencoder_o_test.fit_transform(test1['object'].values)
y_test_location = labelencoder_l_test.fit_transform(test1['location'].values)
df_RNN = df.copy()
df_RNN['action'] = labelencoder_a.fit_transform(df['action'].values)
df_RNN['object'] = labelencoder_o.fit_transform(df['object'].values)
df_RNN['location'] = labelencoder_l.fit_transform(df['location'].values)
tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(df1['transcription'])
sequences = tokenizer.texts_to_sequences(df1['transcription'])
MAXLEN = 7
X = pad_sequences(sequences, maxlen=MAXLEN)
y = df_RNN.iloc[:,2:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state = 42)
main_input = Input(shape=(MAXLEN,), dtype='int32', name='main_input')
x = Embedding(4, 7, input_length=MAXLEN)(main_input)
#x = Dropout(0.1)(x)
x = LSTM(100)(x)
len_list = [len(df.action.unique()), len(df.object.unique()) , len(df.location.unique())]
label_names = ['action','object','location']
output_array = []
metrics_array = {}
loss_array = {}
for i in range(0,3):
    categorical_output = Dense(len_list[i], activation='softmax', name=label_names[i])(x)
    output_array.append(categorical_output)
    metrics_array[label_names[i]] = 'sparse_categorical_accuracy'
    loss_array[label_names[i]] = 'sparse_categorical_crossentropy'
model = Model(inputs=main_input, outputs=output_array)
model.compile(optimizer='adam',
              loss=loss_array,
              metrics = metrics_array)
#For TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
y_train_output=[]
for col in label_names:
    y_train_output.append(y_train[col])
model.fit(X_train, y_train_output,epochs=yaml_data['epochs'], batch_size=yaml_data['batch_size'],verbose=yaml_data['verbose']);
tokenizer = Tokenizer(num_words=4)
tokenizer.fit_on_texts(test1['transcription'])
sequences = tokenizer.texts_to_sequences(test1['transcription'])
MAXLEN = 7
X_test = pad_sequences(sequences, maxlen=MAXLEN)
t = [y_test_action,y_test_object,y_test_location]
preds = model.predict(X_test)
print("F1-Score for Action:")
for i in range(0,len(df.action.unique())):
    print(df.action.unique()[i] + ": ",f1_score(t[0],np.argmax(preds[0],axis=1), average=None)[i])
print("-------------------------------------------------------------------------")
print("F1-Score for Object:")
for i in range(0,len(df.object.unique())):
    print(df.object.unique()[i] + ": ",f1_score(t[1],np.argmax(preds[1],axis=1), average=None)[i])
print("-------------------------------------------------------------------------")
print("F1-Score for Location:")
for i in range(0,len(df.location.unique())):
    print(df.location.unique()[i] + ": ",f1_score(t[2],np.argmax(preds[2],axis=1), average=None)[i])
model.save("model.h5")