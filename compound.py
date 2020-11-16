# -*- coding: utf-8 -*-
#!pip install wandb
#import wandb
#wandb.login()

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Bidirectional, Concatenate
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from keras.layers import Layer
import keras.backend as K
from wandb.keras import WandbCallback
import pickle
from keras.models import load_model
from keras.callbacks import EarlyStopping

"""# Requirements"""

def get_data(path):
  with open(path) as f:
    lines=f.read().splitlines()
    ol=[]
    w1l=[]
    w2l=[]
    for line in lines:
        words=line.split(" ")
        kridant=words[0]
        dhatu=words[2]
        pratya=words[4]
        ol.append(kridant)
        w1l.append(dhatu)
        w2l.append(pratya)

  return w1l,w2l,ol

def get_texts(file_path,test_split=0.2,random_state=1):
  input_texts = []
  target_texts = []
  #w1l, w2l, ol = pdp.get_xy_data("Data/kridantaList.txt")
  # Reading directly from converted file
  #Satf~ .... total: 324 correct: 179 ....accuracy: 55.24691358024691
  #SAnac .... total: 200 correct: 116 ....accuracy: 57.99999999999999
  w1l,w2l,ol=get_data(file_path)

  print("Sandhi dataset created")
  ct=0
  for i in range(len(w1l)):
    if(w2l[i]=="Satf~" or w2l[i]=="SAnac"):
      ct+=1
      continue
    input_text = w1l[i] + '+' + w2l[i]
    target_text = ol[i]
    # We use "&" as the "start sequence" character for the targets, and "$" as "end sequence" character.
    target_text = '&' + target_text + '$'
    input_texts.append(input_text)
    target_texts.append(target_text)
  print(ct)
  if(test_split!=0):
    X_train, X_test, Y_train, Y_test = train_test_split(input_texts, target_texts, test_size=test_split, random_state=random_state)
    return X_train,X_test,Y_train,Y_test
  return input_texts,target_texts

def get_d(X,tokens):
  for sentence in X:
    for char in sentence:
      tokens.add(char)
  return tokens

def get_trained_model(architecture,X_train,Y_train,X_test=None,Y_test=None,latent_dim=32,batch_size=64,epochs=70,validation_split=0.2,verbose=1,use_wandb=False,model=None):
  re=True
  if(model==None):
    re=False
    model=Translator()
    model.latent_dim=latent_dim
    # getting model dictionary
    input_texts=[x for x in X_train]
    target_texts=[y for y in Y_train]
    if(X_test!=None and Y_test!=None):
      input_texts.extend(X_test)
      target_texts.extend(Y_test)
    characters=set()
    characters=get_d(input_texts,characters)
    characters=get_d(target_texts,characters) 

    model.max_encoder_seq_length = max([len(txt) for txt in input_texts])
    model.max_decoder_seq_length = max([len(txt) for txt in target_texts])

    # Using '*' for padding 
    characters.add('*')

    characters = sorted(list(characters))
    model.num_tokens = len(characters)

    print('Number of samples:', len(input_texts))
    print('Number of unique tokens:', model.num_tokens)
    print('Max sequence length for inputs:', model.max_encoder_seq_length)
    print('Max sequence length for outputs:', model.max_decoder_seq_length)

    model.token_index = dict([(char, i) for i, char in enumerate(characters)])
    model.reverse_target_char_index = dict((i, char) for char, i in model.token_index.items())


  encoder_input_data = np.zeros((len(X_train), model.max_encoder_seq_length,model.num_tokens), dtype='float32')
  decoder_input_data = np.zeros((len(X_train), model.max_decoder_seq_length, model.num_tokens), dtype='float32')
  decoder_target_data = np.zeros((len(X_train), model.max_decoder_seq_length, model.num_tokens), dtype='float32')

  for i, (input_text, target_text) in enumerate(zip(X_train, Y_train)):
    for t, char in enumerate(input_text):
      encoder_input_data[i, t, model.token_index[char]] = 1.
    encoder_input_data[i, t + 1:, model.token_index['*']] = 1.
    for t, char in enumerate(target_text):
      # decoder_target_data is ahead of decoder_input_data by one timestep
      decoder_input_data[i, t, model.token_index[char]] = 1.
      if t > 0:
        # decoder_target_data will be ahead by one timestep
        # and will not include the start character.
        decoder_target_data[i, t - 1, model.token_index[char]] = 1.
    decoder_input_data[i, t + 1:, model.token_index['*']] = 1.
    decoder_target_data[i, t:, model.token_index['*']] = 1.
  if(re==True):
    model.model=train(model.model,encoder_input_data,decoder_input_data,decoder_target_data,batch_size,epochs,validation_split,verbose,use_wandb,re=re)
    return model
  # train the model on text data
  model.model,model.encoder_model,model.decoder_model=architecture(model.latent_dim,model.num_tokens)
  model.model=train(model.model,encoder_input_data,decoder_input_data,decoder_target_data,batch_size,epochs,validation_split,verbose,use_wandb,re=re)
  
  #save the best model and last model before returning
  return model

def use_wandb(project_name,run_name,batch_size,epochs,validation_split,latent_dim):
    wandb.init(project=project_name,name=run_name)
    config=wandb.config
    config.epochs=epochs
    config.batch_size=batch_size
    config.validation_split=validation_split
    config.latent_dim=latent_dim

def train(model,encoder_input_data,decoder_input_data,decoder_target_data,batch_size,epochs,validation_split,verbose,use_wandb=False,re=False):
  if(re==False):
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
  if(use_wandb==True):
    #print("Running:",run_name)
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,patience=4)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=validation_split,verbose=verbose,callbacks=[WandbCallback(),es])
  else:
    es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1,patience=4)
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=validation_split,verbose=verbose,callbacks=[es])
  return model


def normal_testing(X,Y,model,verbose=1,use_wandb=False,log_as="test_passed"):
  #file1 = open("Incorrect_predictions.txt", "w")    
  predictions=model.get_predictions(X)
  total=len(predictions)
  passed=0
  true=[a[1:-1] for a in Y]
  #passed=np.sum([1 for i in range(total) if(predictions[i]==true[i])])
  for i in range(total):
    if(predictions[i]==true[i]):
      passed+=1
    else:
      if(verbose==1):
        print(str(i)+'/'+str(total))
        print('-')
        print('Input sentence:   ', X[i])
        print('Decoded sentence: ', predictions[i])
        print('Expected sentence:', true[i])
  if(use_wandb==True):
    #wandb.log({log_as:str(passed)+"/"+str(total)})
    wandb.log({log_as:passed/total})
  print(log_as+":"+str(passed)+'/'+str(total))
  return passed,total


def K_Fold_testing(X,Y,model_architecture,k,latent_dim=32,batch_size=64,epochs=70,validation_split=0,verbose=0,use_wandb=False,run_name="None",log_as="test_passed"):
  acc=[]
  train_acc=[]
  if(use_wandb):
    wandb.config.num_k=k   
  kf=KFold(n_splits=k,random_state=1,shuffle=True)
  kf.get_n_splits(X,Y)
  i=1
  for train_index,test_index in kf.split(X):
    x_train=list(np.array(X)[train_index.astype(int)])
    x_test=list(np.array(X)[test_index.astype(int)])
    y_train=list(np.array(Y)[train_index.astype(int)])
    y_test=list(np.array(Y)[test_index.astype(int)])
    print("split_number:",i)
    model=get_trained_model(architecture=model_architecture,X_train=x_train,Y_train=y_train,X_test=x_test,Y_test=y_test,latent_dim=latent_dim,batch_size=batch_size,epochs=epochs,validation_split=validation_split,verbose=verbose,use_wandb=use_wandb)
    train_passed,train_out_of=normal_testing(x_train,y_train,model,verbose=0,use_wandb=use_wandb,log_as="train_passed")
    passed,outof=normal_testing(x_test,y_test,model,verbose=0,use_wandb=use_wandb,log_as="test_passed")
    train_acc.append(train_passed/train_out_of)
    acc.append(passed/outof)
    print("------------")
    i+=1
  kf_acc=np.sum(acc)/k
  kf_train_acc=np.sum(train_acc)/k
  if(use_wandb):
    wandb.log({"KFold_accuracy":kf_acc})
    wandb.log({"KFold_train_accuracy":kf_train_acc})
  print("KFold_train_acc=",kf_train_acc)
  print("KFold_test_acc=",kf_acc)

class Translator:
  def init(self):
    self.token_index=None
    self.num_tokens=0
    self.max_encoder_seq_length=0
    self.max_decoder_seq_length=0
    self.model=None
    self.encoder_model=None
    self.decoder_model=None
    self.reverse_target_char_index=None

  def dictionary_info(self):
    
    print("num_tokens",self.num_tokens)
    print("max_encoder_seq_length",self.max_encoder_seq_length)
    print("max_decoder_seq_length",self.max_decoder_seq_length)
    print("token_index")
    print(self.token_index)
    print("reverse_target_char_index")
    print(self.reverse_target_char_index)

  def vectorize(self,X):
    encoder_input_data = np.zeros((len(X), self.max_encoder_seq_length, self.num_tokens), dtype='float32')
    for i, input_text in enumerate(X):
      for t, char in enumerate(input_text):
        if char not in self.token_index:
          continue
        encoder_input_data[i, t, self.token_index[char]] = 1.
      encoder_input_data[i, t + 1:, self.token_index['*']] = 1.
    return encoder_input_data

  def get_predictions(self,X):
    ans=[]   
    encoder_input_data=self.vectorize(X)
    for seq_index in range(len(encoder_input_data)):
      input_seq = encoder_input_data[seq_index: seq_index + 1]
      decoded_sentence = self.decode_sequence(input_seq)
      #print(decoded_sentence,input_seq)
      decoded_sentence = decoded_sentence.strip()
      decoded_sentence = decoded_sentence.strip('$')
      ans.append(decoded_sentence)
    return ans

  def decode_sequence(self,input_seq):
    # Encode the input as state vectors.
    states_value = self.encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, self.num_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, self.token_index['&']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
      output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
      # Sample a token
      sampled_token_index = np.argmax(output_tokens[0, -1, :])
      sampled_char = self.reverse_target_char_index[sampled_token_index]
      decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
      if (sampled_char == '$' or
        len(decoded_sentence) > self.max_decoder_seq_length):
          stop_condition = True

        # Update the target sequence (of length 1).
      target_seq = np.zeros((1, 1, self.num_tokens))
      target_seq[0, 0, sampled_token_index] = 1.

          # Update states
      states_value = [h, c]

    return decoded_sentence

# use maximum seq length for input before using i.e 18 for taddhita and 17 for kridant
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(18,1),initializer="zeros")                # Change the sequence length before using. It depends on the dataset and for us it was (16,1) for splitting and (17,1) for synthesis 
        super(attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        #print(np.shape(et))
        #print("....shape et")
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(attention,self).get_config()


def get_model_attention(latent_dim,num_tokens):
  
  encoder_inputs = Input(shape=(None,num_tokens))
  encoder = Bidirectional(LSTM(latent_dim,return_sequences=True,return_state=True,recurrent_dropout=0.2))
  att_in,forward_h, forward_c, backward_h, backward_c=encoder(encoder_inputs)
  att_out=attention()(att_in)
    
  state_c = Concatenate()([forward_c, backward_c])
    
  encoder_states=[att_out,state_c]
    
  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = Input(shape=(None, num_tokens))
  # We set up our decoder to return full output sequences,
  # and to return internal states as well. We don't use the
  # return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True,recurrent_dropout=0.2)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
  decoder_dense = Dense(num_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
      
  encoder_model = Model(encoder_inputs, encoder_states)
    
  decoder_state_input_h = Input(shape=(latent_dim*2,))
  decoder_state_input_c = Input(shape=(latent_dim*2,))
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
  decoder_states = [state_h, state_c]
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

  return model,encoder_model,decoder_model


def get_model(latent_dim,num_tokens):
  
  encoder_inputs = Input(shape=(None, num_tokens))
  encoder = Bidirectional(LSTM(latent_dim, return_state=True,recurrent_dropout=0.2))
  encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
  state_h = Concatenate()([forward_h, backward_h])
  state_c = Concatenate()([forward_c, backward_c])

  # We discard `encoder_outputs` and only keep the states.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  decoder_inputs = Input(shape=(None, num_tokens))

  # We set up our decoder to return full output sequences,
  # and to return internal states as well. We don't use the
  # return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True,recurrent_dropout=0.2)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
  decoder_dense = Dense(num_tokens, activation='softmax')
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
      
  encoder_model = Model(encoder_inputs, encoder_states)
    
  decoder_state_input_h = Input(shape=(latent_dim*2,))
  decoder_state_input_c = Input(shape=(latent_dim*2,))
  decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
  decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
  decoder_states = [state_h, state_c]
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

  return model,encoder_model,decoder_model

def save_model(filename,model_obj):
  mod=model_obj
  fileobj=open(filename,'wb')
  pickle.dump(mod,fileobj)
    
def restore_model(filename):                                                
  return pickle.load(open(filename,'rb'))

#translator.model.save("a.h")

#translator.model=load_model("a.h")

"""# Training"""

X_train, X_test, Y_train, Y_test=get_texts("/content/black_and_yellow_encoded.txt",random_state=1)

# can also use get_model instead of get_attention model to run the model without attention
translator=get_trained_model(get_model_attention,X_train,Y_train,X_test,Y_test,epochs=70,validation_split=0.1,batch_size=32,latent_dim=64,use_wandb=False)

normal_testing(X_test,Y_test,model=translator,verbose=0,use_wandb=False)

normal_testing(X_train,Y_train,model=translator,verbose=0,use_wandb=False,log_as="train_passed")



"""## Narrowing epoch range"""

#translator=get_trained_model(get_model,X_train,Y_train,X_test,Y_test,epochs=10,verbose=2,batch_size=32,latent_dim=64)
#normal_testing(X_test,Y_test,model=translator,verbose=0)
#for i in range(9):
#  print("Increasing num of epochs by 10: total epochs=",10*(i+2))
#  translator=get_trained_model(get_model_attention,X_train,Y_train,X_test,Y_test,epochs=10,model=translator,verbose=2,batch_size=32,latent_dim=64)
#  a,b=normal_testing(X_test,Y_test,model=translator,verbose=0)
#  print("total_epochs=",10*(i+2),"test_acc=",(a*100)/b)

#normal_testing(X_test,Y_test,translator)

#translator=get_trained_model(get_model_attention,X_train,Y_train,X_test,Y_test,epochs=5,model=translator)

"""## K-Fold"""

#use_wandb("kridanta_synthesis","Swapped_parameters_Regularized_KFold_with_attention_1",batch_size=32,epochs=40,validation_split=0.1,latent_dim=64)
#X,Y=get_texts("/content/log.txt",test_split=0)
#K_Fold_testing(X,Y,get_model_attention,5,verbose=1,validation_split=0.1,use_wandb=True,latent_dim=64,batch_size=32,epochs=40)