# Text-Classifier-NLP
* The Data consists of transcription (Action or Process describing something) fetched from the audio files. I extracted action, object and location from the text using Natural Language Processing.
* Below are the steps:
1.	Checking the number of GPUs available and if they are more than 1, then utilizing those GPUs for processing.
2.	Parsing Argument for passing the model parameters using the config.yaml file. 
3. Data Preprocessing:
    * Lower case the text
    * Removing punctuations
    * Removing Stop words: For that I used the nltk library that has over 7000 stop words in different languages. From those I removed “on” and “off” as Its shows some action so its important part of the text to predict the action.
4.	Label Encoding:
    * Encoding Categorical variables into integers using LabelEncoder from sklearn. It is a important step in order to make model understand the dependent variables.
5.	Tokenizing:
    * Tokenize the text and sequence it using some numerical representation using keras.
6.	Splitting Dataset: Splitting the Dataset into train and test.
7.	Model Building: used RNN.
8.	Storing the logs into tensorboard.
9.	Saving the model in the h5 format.




	Steps for Training and testing the model:
In the terminal write:
        `python train.py --config "config.yaml"`
* You will get a model.h5 file.
Next to Test data, write:
        `python test.py --testcsv "test.csv"`
* Output will be predicted that shows the data frame consisting of action, object and location of the corresponding texts.

