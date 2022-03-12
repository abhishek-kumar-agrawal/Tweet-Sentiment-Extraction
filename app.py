# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:41:41 2022

@author: HP
"""

import streamlit as st
import flask
import pandas as pd
from joblib import dump, load
import numpy as np

from tqdm import tqdm
tqdm.pandas()
from keras.models import model_from_json 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


#uploaded_file = st.file_uploader('Choose a file')
#if uploaded_file is not None:
#    inputs = pd.read_csv(uploaded_file)
    

with open('tokenizer_t.pkl','rb') as f:
    tokenizer_text = pickle.load(f)

with open('tokenizer_s.pkl','rb') as f:
    tokenizer_sentiment = pickle.load(f)

# Loading JSON file
from keras.models import model_from_json  
json_file = open("model_final_1.json", 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
  
# Loading weights
loaded_model.load_weights("model_final_1.h5")

def find_sel_text(x):
    """ this function will convert the index to text"""
    txt , start , end = x[0],x[1],x[2]

    end=end+1
    txt = txt.split()
    sel_text = txt[start:end]
    sel_text = " ".join(sel_text)
    return sel_text

def predict(X):
    '''This function takes a text(string) and sentiment as input and 
       returns selected_text(keywords) as output'''
    df_text = X['text'].values
    df_sentiment = X['sentiment'].values

    df_text = tokenizer_text.texts_to_sequences(df_text)
    df_text = pad_sequences(df_text,32,padding='post')

    df_sentiment = tokenizer_sentiment.texts_to_sequences(df_sentiment)
    df_sentiment = pad_sequences(df_sentiment,1,padding='post')
    prediction = loaded_model.predict([df_text,df_sentiment])
    X['start'],X['end'] = (abs(prediction[:,0])),(abs(prediction[:,1]))
    X['start'] = X['start'].astype('int')
    X['end'] = X['end'].astype('int')
    X['predicted_text'] = X[['text','start','end']].progress_apply(lambda i : find_sel_text(i),axis=1)
    X = X.drop(['start','end'],axis=1)
    #return X['predicted_text'].values
    #return X.predicted_text.values
    return X.iat[0,2]

def main():
    st.set_page_config(page_title="Tweet Sentiment Phrase Extraction", 
                       page_icon=":robot_face:",
                       layout="wide",
                       )
    st.markdown("<h2 style='text-align: center; color:grey;'>Tweet Sentiment Phrase Extraction &#129302;</h2>", unsafe_allow_html=True)
    st.text('')
    #st.markdown("<h4 style='text-align: center; color:grey;'>Sentiment &#129302;</h4>", unsafe_allow_html=True)
    #st.text('')
    st.markdown("<h4 style='text-align: center; color:grey;'>Original Text &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    input_text = st.text_area("Enter text", max_chars=500, height=150)
    st.markdown("<h4 style='text-align: center; color:grey;'>Sentiment &#129302;</h4>", unsafe_allow_html=True)
    st.text('')
    input_sentiment = st.text_area("Enter sentiment", max_chars=500, height=150)
    inputs = pd.DataFrame([[input_text,input_sentiment]],columns=['text', 'sentiment'],dtype='str')
    st.markdown(f'<h4 style="text-align: left; color:#F63366; font-size:28px;">Supported Phrase</h4>', unsafe_allow_html=True)
    st.text('')
    if st.button("Predict"):
        output = predict(inputs)
        st.write(output)
 
if __name__=='__main__':
    main()