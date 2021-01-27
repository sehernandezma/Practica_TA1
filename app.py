#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import json
import requests
from flask import Flask, request

# Adicional
"""Practica1_chatbot.ipynb
Limpa el Corpus
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from pathlib import Path
import string
import unidecode
import nltk
from nltk.corpus import stopwords

np.random.seed(seed=0)
import tensorflow-cpu as tf
tf.random.set_seed(0)
"""
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
root_dir = "/content/drive/My Drive/"
base_dir = root_dir + 'practica1_topicos1/'

model_dict = base_dir + 'Modelos descargados/'


path=Path(base_dir)
print(path)
"""

nltk.download('stopwords')

stop_words = stopwords.words('spanish')

def limpiezatexto(text):
  text = text.replace('-',' ')
  text = text.replace('/',' ')
  text = text.translate(str.maketrans('', '', string.punctuation))
  text = text.lower()
  text = ' '.join([word for word in text.split() if word not in stop_words])
  text = unidecode.unidecode(text)
  return text

import re
import random
#with open(path/'preguntas_Literatura.txt',mode='r',encoding='utf-8') as f:
with open('preguntas_Literatura.txt',mode='r',encoding='utf-8') as f:
  lines=f.read().split('\n')
#with open(path/'respuestas_Literatura.txt',mode='r',encoding='utf-8') as f:
with open('respuestas_Literatura.txt',mode='r',encoding='utf-8') as f:
  lines2=f.read().split('\n')
  lines=[re.sub(r"\[\w+\]",'',line) for line in lines]
  lines=[" ".join(re.findall(r"\w+",line)) for line in lines]
  lines2=[re.sub(r"\[\w+\]",'',line) for line in lines2]
  lines2=[" ".join(re.findall(r"\w+",line)) for line in lines2]
  lines = list(map(limpiezatexto, lines))
  lines2 = list(map(limpiezatexto, lines2))
  pairs=list(zip(lines,lines2))
  #random.shuffle(pairs)

import numpy as np
input_docs = []
target_docs = []
input_tokens = set()
target_tokens = set()
for line in pairs[:]:
  input_doc, target_doc = line[0], line[1]
  # Appending each input sentence to input_docs
  input_docs.append(input_doc)
  # Splitting words from punctuation  
  target_doc = " ".join(re.findall(r"[\w']+|[^\s\w]", target_doc))
  # Redefine target_doc below and append it to target_docs
  target_doc = '<START> ' + target_doc + ' <END>'
  target_docs.append(target_doc)
  
  # Now we split up each sentence into words and add each unique word to our vocabulary set
  for token in re.findall(r"[\w']+|[^\s\w]", input_doc):
    if token not in input_tokens:
      input_tokens.add(token)
  for token in target_doc.split():
    if token not in target_tokens:
      target_tokens.add(token)
input_tokens = sorted(list(input_tokens))
target_tokens = sorted(list(target_tokens))
num_encoder_tokens = len(input_tokens)
num_decoder_tokens = len(target_tokens)

input_features_dict = dict(
    [(token, i) for i, token in enumerate(input_tokens)])
target_features_dict = dict(
    [(token, i) for i, token in enumerate(target_tokens)])
reverse_input_features_dict = dict(
    (i, token) for token, i in input_features_dict.items())
reverse_target_features_dict = dict(
    (i, token) for token, i in target_features_dict.items())

#Maximum length of sentences in input and target documents
max_encoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", input_doc)) for input_doc in input_docs])
max_decoder_seq_length = max([len(re.findall(r"[\w']+|[^\s\w]", target_doc)) for target_doc in target_docs])
encoder_input_data = np.zeros(
    (len(input_docs), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_docs), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
for line, (input_doc, target_doc) in enumerate(zip(input_docs, target_docs)):
    for timestep, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_doc)):
        #Assign 1. for the current line, timestep, & word in encoder_input_data
        encoder_input_data[line, timestep, input_features_dict[token]] = 1.
    
    for timestep, token in enumerate(target_doc.split()):
        decoder_input_data[line, timestep, target_features_dict[token]] = 1.
        if timestep > 0:
            decoder_target_data[line, timestep - 1, target_features_dict[token]] = 1.

from tensorflow-cpu import keras
from keras.layers import Input, LSTM, Dense
from keras.models import Model
#Dimensionality
dimensionality = 256
#The batch size and number of epochs
batch_size = 5
epochs = 300
#Encoder
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder_lstm = LSTM(dimensionality, return_state=True)
encoder_outputs, state_hidden, state_cell = encoder_lstm(encoder_inputs)
encoder_states = [state_hidden, state_cell]
#Decoder
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(dimensionality, return_sequences=True, return_state=True)
decoder_outputs, decoder_state_hidden, decoder_state_cell = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

#Model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

from keras.models import load_model
training_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
#training_model.load_weights(model_dict + 'training_model Test.h5')# Cambiar path 
training_model.load_weights('Models/training_model Test.h5')
encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

latent_dim = 256
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

def decode_response(test_input):
  #Getting the output states to pass into the decoder
  states_value = encoder_model.predict(test_input)
  #Generating empty target sequence of length 1
  target_seq = np.zeros((1, 1, num_decoder_tokens))
  #Setting the first token of target sequence with the start token
  target_seq[0, 0, target_features_dict['<START>']] = 1.
  
  #A variable to store our response word by word
  decoded_sentence = ''
  
  stop_condition = False
  while not stop_condition:
      #Predicting output tokens with probabilities and states
    output_tokens, hidden_state, cell_state = decoder_model.predict([target_seq] + states_value)
  #Choosing the one with highest probability
    sampled_token_index = np.argmax(output_tokens[0, -1, :])
    sampled_token = reverse_target_features_dict[sampled_token_index]
    decoded_sentence += " " + sampled_token
  #Stop if hit max length or found the stop token
    if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
      stop_condition = True
  #Update the target sequence
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, sampled_token_index] = 1.
    #Update states
    states_value = [hidden_state, cell_state]
  return decoded_sentence

class ChatBot:
  negative_responses = ("no", "nope", "nah", "naw", "not a chance", "sorry")
  exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later", "stop", "salir", "pausar", "chao", "detener", "hasta luego")
#Method to start the conversation
  def start_chat(self):
    user_response = input("Hola, soy un chatbot listo para responder tus preguntas de literura, ¿En qué te puedo ayudar?\n")
    
    if user_response in self.negative_responses:
      print("Que tengas un bonito día!")
      return
    self.chat(user_response)
#Method to handle the conversation
  def chat(self, reply):
    while not self.make_exit(reply):
      reply = input(self.generate_response(reply)+"\n")
    
  #Method to convert user input into a matrix
  def string_to_matrix(self, user_input):
    tokens = re.findall(r"[\w']+|[^\s\w]", user_input)
    user_input_matrix = np.zeros(
      (1, max_encoder_seq_length, num_encoder_tokens),
      dtype='float32')
    for timestep, token in enumerate(tokens):
      if token in input_features_dict:
        user_input_matrix[0, timestep, input_features_dict[token]] = 1.
    return user_input_matrix
  
  #Method that will create a response using seq2seq model we built
  def generate_response(self, user_input):
    input_matrix = self.string_to_matrix(user_input)
    chatbot_response = decode_response(input_matrix)
    #Remove <START> and <END> tokens from chatbot_response
    chatbot_response = chatbot_response.replace("<START>",'')
    chatbot_response = chatbot_response.replace("<END>",'')
    return chatbot_response
#Method to check for exit commands
  def make_exit(self, reply):
    for exit_command in self.exit_commands:
      if exit_command in reply:
        print("Ok, Que tengas un bonito día!")
        return True
    return False
  
chatbot = ChatBot()

#Hasta aquí

app = Flask(__name__)


@app.route('/', methods=['GET'])
def verify():

    # cuando el endpoint este registrado como webhook, debe mandar de vuelta
    # el valor de 'hub.challenge' que recibe en los argumentos de la llamada

    if request.args.get('hub.mode') == 'subscribe' \
        and request.args.get('hub.challenge'):
        if not request.args.get('hub.verify_token') \
            == os.environ['VERIFY_TOKEN']:
            return ('Verification token mismatch', 403)
        return (request.args['hub.challenge'], 200)

    return ('Hello world', 200)


@app.route('/', methods=['POST'])
def webhook():

    # endpoint para procesar los mensajes que llegan

    data = request.get_json()

    # log(data)  # logging, no necesario en produccion

    inteligente = True

    if data['object'] == 'page':

        for entry in data['entry']:
            for messaging_event in entry['messaging']:

                if messaging_event.get('message'):  # alguien envia un mensaje

                    sender_id = messaging_event['sender']['id']  # el facebook ID de la persona enviando el mensaje
                    recipient_id = messaging_event['recipient']['id']  # el facebook ID de la pagina que recibe (tu pagina)
                    message_text = messaging_event['message']['text']  # el texto del mensaje

## Parte modificada
                    respuesta = ChatBot().generate_response(message_text)

                    send_message(sender_id, respuesta)
## Fin parte modificada

                if messaging_event.get('delivery'):  # confirmacion de delivery
                    pass

                if messaging_event.get('optin'):  # confirmacion de optin
                    pass

                if messaging_event.get('postback'):  # evento cuando usuario hace click en botones
                    pass

    return ('ok', 200)


def send_message(recipient_id, message_text):

    # log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    params = {'access_token': os.environ['PAGE_ACCESS_TOKEN']}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps({'recipient': {'id': recipient_id},
                      'message': {'text': message_text}})

    r = requests.post('https://graph.facebook.com/v2.6/me/messages',
                      params=params, headers=headers, data=data)
    if r.status_code != 200:
        log(r.status_code)
        log(r.text)


def log(message):  # funcion de logging para heroku
    print(str(message))
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)