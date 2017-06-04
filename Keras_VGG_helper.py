
# coding: utf-8

# # *Planet: Understanding the Amazon from Space* challenge
# 
# This notebook will show you how to do some basic manipulation of the images and label files.

# In[62]:

import sys
import os
import subprocess
import sys
import time

from six import string_types

# Make sure you have all of these packages installed, e.g. via pip
import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io
from skimage.transform import rescale


# ## Setup
# Set `PLANET_KAGGLE_ROOT` to the directory where you've downloaded the TIFF and JPEG zip files, and accompanying CSVs.


# PLANET_KAGGLE_ROOT = os.path.abspath("C:/forest/")
# PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-tif-v2/')
# PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
# MODEL_SAVE_FILE_NAME = 'ver_3_forest-vgg.keras'

#PLANET_KAGGLE_ROOT = os.path.abspath("C:/Kaggle")
#PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'planet-amazon-deforestation-master','train-tif-v2/')
#PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'planet-amazon-deforestation-master', 'train_v2.csv')
#MODEL_SAVE_FILE_NAME = 'ver_3_forest-vgg.keras'
#
#print(PLANET_KAGGLE_JPEG_DIR)
#
#
#assert os.path.exists(PLANET_KAGGLE_ROOT)
#assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
#assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)



land_labels = ['primary', 'agriculture', 'water', 'cultivation', 'habitation']
rare_labels = ['slash_burn','conventional_mine','bare_ground','artisinal_mine','blooming','selective_logging','blow_down']
weather_labels = ['clear', 'partly_cloudy', 'haze', 'cloudy']


def load_image(filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    #for dirname in os.listdir(PLANET_KAGGLE_ROOT):
    path = os.path.abspath(os.path.join(PLANET_KAGGLE_JPEG_DIR, filename))
    if os.path.exists(path):
        #print('Found image {}'.format(path))
        return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
    
def load_single_image(path, filename):
    '''Look through the directory tree to find the image you specified
    (e.g. train_10.tif vs. train_10.jpg)'''
    path = os.path.abspath(os.path.join(path, filename))
    if os.path.exists(path):
        #print('Found image {}'.format(path))
        return io.imread(path)
    # if you reach this line, you didn't find the image you're looking for
    print('Load failed: could not find image {}'.format(path))
    
def sample_to_fname(sample_df, row_idx, suffix='tif'):
    '''Given a dataframe of sampled images, get the
    corresponding filename.'''
    fname = sample_df.get_value(sample_df.index[row_idx], 'image_name')
    return '{}.{}'.format(fname, suffix)

# Grava no arquivo de log
def write_to_file(filename, text):
    f = open(filename, 'a')
    f.write('Data: {}: {} \n'.format(time.ctime(), text))
    f.close()


#  ### Add onehot features for every label

def load_labels(labels_dir):
    #Carrega labels e adiciona one hot features
    
    print(labels_dir)

    labels_df = pd.read_csv(labels_dir)
    
    from collections import Counter
    total_counts = Counter()
    for labels in labels_df['tags']:
        total_counts.update(labels.split())
    
    vocab = sorted(total_counts, key=total_counts.get, reverse=True)
    
    # Add onehot features for every label
    for label in vocab:
        labels_df[label] = labels_df['tags'].apply(lambda x: 1 if label in x.split(' ') else 0)
        
    return labels_df


def subsample(df, n_sample):
    df_sample = pd.DataFrame()
    columns = list(df)

    i=0
    df_factor  = df.groupby(['tags'])['tags'].sum().sort_values().keys()
    for tags in df_factor:
            
            if tags == 'cloudy':
                sample = n_sample * 50
            else:
                sample = n_sample
                            
            
            df_s = df[df['tags'] == tags].sample(sample, replace=True)
            df_sample = df_sample.append(df_s)
            
    return df_sample[columns]
    


# ### loading multiple JPEG files, normalize and categorize labels


def load_images(img_dir,labels_df, img_fim=100, img_type='jpg', img_ini=0, img_rescale=1.0, sample=0, debug = False):
    #Carrega as imagens e os labels

    X = []
    y = []
    
    i = img_ini
    
    columns = list(labels_df)

    # Sub-sample
    if sample != 0:
        labels_df = subsample(labels_df, sample)[columns]
        img_ini = 0
        img_fim = len(labels_df) - 1
    
    #Variavel para carregar as imagens efetivamente carregadas
    labels_res = pd.DataFrame()
    
    print('Processando imagens '+ str(img_ini), str(img_fim) )

    write_to_file('log_run.log', 'Processando imagens ' + str(img_ini) + ' ' + str(img_fim))
    
    for i in range(img_ini, img_fim):                        
                    
        try: 
            img_x = load_image(img_dir + labels_df.iloc[i].values[0] + '.' + img_type)
            
            if img_rescale != 1.0:    
                img_x = rescale(img_x, img_rescale, mode='reflect')
                
            y.append(labels_df.iloc[i].values[2:])    
            X.append(img_x/255)
            
            labels_res = labels_res.append(labels_df.iloc[i])
                
        except:
            if debug: print('ERRO LEITURA IMAGEM:',img_dir + labels_df.iloc[i].values[0] + '.' + img_type)
            pass
        
        
                
        if i >= len(labels_df) - 1:
            break
        
        
    X = np.array(X, dtype='float32')

    one_hot_encode = np.array(y, dtype='float32')

    return X, one_hot_encode


def load_test_images(img_dir, jpg_list, labels_df, img_ini=0, img_fim=100):
    #Carrega as imagens e os labels
    
    X = []
    file_names = []
        
    i = img_ini
    
    print('Processando imagens '+ str(img_ini), str(img_fim) )
    
    for i in range(img_ini, img_fim):                        
                    
        try: 
            img_x = load_single_image(img_dir, jpg_list[i])                
            file_names.append(jpg_list[i])    
            X.append(img_x/255)
        
        except:
            print('ERRO LEITURA IMAGEM:',jpg_list[i])
            pass        
        
                
        if i >= len(jpg_list) - 1:
            break
        
        
    X = np.array(X, dtype='float32')

    return X, file_names



def get_img_shape(rescale_img = 0.0):
    norm, one_hot = load_images(PLANET_KAGGLE_JPEG_DIR, PLANET_KAGGLE_LABEL_CSV,
                                img_ini=10, img_fim=11, 
                                img_type='tif', debug=True,
                                img_rescale=rescale_img)
    
    return [None, norm[0][:,:,:].shape[0], norm[0][:,:,:].shape[1], norm[0][:,:,:].shape[2]]
    



def load_data(img_dir,labels_df, img_ini=0, img_type='tif', batch_size=0, res_img=1.0, sample=0):
    #Cria os batches de images para serem processados
    
    #Separar 10% do batch para teste
                
    img_fim = img_ini + batch_size - 1
    
    # print(img_ini, img_fim)
        
    X_features, Y_features = load_images(img_dir,labels_df, img_fim, img_type, img_ini, img_rescale = res_img, sample=sample)
    
    if batch_size == 0:     
        train_size = len(Y_features)
    else:
        train_size = int(0.9 * batch_size)
    
    
    return (X_features[:train_size], Y_features[:train_size]), (X_features[:-train_size], Y_features[:-train_size])
        

def weight_dict(range_qtd):

    labels_df = load_labels(PLANET_KAGGLE_LABEL_CSV)  

    labels = labels_df.columns[2:]     

    weight_list = (labels_df[labels].sum().max() / labels_df[labels].sum())

    weight_d=[]
    weight_broadcast = []
    i = 0
    for v in weight_list:
        weight_d.append(v)

    # for i in range(range_qtd):
    #     weight_broadcast.append(weight_d)
    
    weigh_np = np.array(weight_d)

    return weigh_np

# ### Modelo Keras


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K

from tensorflow import reset_default_graph

def model_keras():
  
    global model

    model = Sequential()

    model.add(Conv2D(64, 3, padding='same', input_shape=(256,256,4),activation='relu', name='conv1_1_in'))
    model.add(Conv2D(64, 3, activation='relu', name='conv1_2_in'))    
    model.add(MaxPooling2D(pool_size=2, strides=2, name='pool1'))

    model.add(Conv2D(128, 3, activation='relu', name='conv2_1_in'))    
    model.add(Conv2D(128, 3, activation='relu', name='conv2_2_in'))    
    model.add(MaxPooling2D(pool_size=2, strides=2, name='pool2'))

    model.add(Conv2D(256, 3, activation='relu', name='conv3_1'))
    model.add(Conv2D(256, 3, activation='relu', name='conv3_2'))
    model.add(Conv2D(256, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D(pool_size=2, strides=2, name='pool3'))

    model.add(Conv2D(512, 3, activation='relu', name='conv4_1'))
    model.add(Conv2D(512, 3, activation='relu', name='conv4_2'))
    model.add(Conv2D(512, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D(pool_size=2, strides=2, name='pool4'))

    model.add(ZeroPadding2D(padding=(1, 1)))
    model.add(Conv2D(512, 3, activation='relu', name='conv5_1'))
    model.add(Conv2D(512, 3, activation='relu', name='conv5_2'))
    model.add(Conv2D(512, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D(pool_size=2, strides=2, name='pool5'))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(Dropout(0.5))

    model.add(Dense(17, activation='sigmoid', name='fc3_out'))

    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    # Let's train the model using RMSprop
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


def Contiuar_treinamento():
    #Continuar treinamento já existente    
    model = keras.models.load_model(MODEL_SAVE_FILE_NAME)

    return model

def class_weights(Y):
    #Calcula class weights para classes desbalanceadas
    weight_list = np.sum(Y, axis=0) / np.sum(Y, axis=0).max() 
     
    weights = {}
    i = 0
    for w in weight_list:
        if w != 0:
            weights[i] = int(1/w)
        else:
            weights[i] = 1
        i+=1
        
    return weights
    

# ### Treinando em batches

def batch_training(p_batch_size_load=100, p_epochs=50, p_batch_size_keras=50):    

    model = model_keras()
    
    #Carrega labels e lista de imagens e one hot encoded labels
    
    labels_df = load_labels(PLANET_KAGGLE_LABEL_CSV)
    
    #continuar a partir de 5000
    idx=0
    while idx < 40000:
    
        #Carrega as imagens
        (X, Y), (X_test, Y_test) = load_data(PLANET_KAGGLE_JPEG_DIR,labels_df, 
                                             img_ini=idx, batch_size = p_batch_size_load)
        #Cálculo de classes desbalanceadas
        weights = class_weights(Y)
            
        #Treina o modelo
        history = model.fit(X, Y, 
                            epochs=p_epochs,
                            batch_size=p_batch_size_keras, 
                            class_weight = weights,
                            validation_data=(X_test, Y_test))
        
        
        idx+= p_batch_size_load
          
        #Salva o modelo a cada iteração
        model.save(MODEL_SAVE_FILE_NAME) 
    
# ### Treinando em batches

def batch_classify(model_file, sub_file, img_dir, labels_file, p_batch_size_load=100, p_batch_size_keras=50):    
    
    jpg_list = os.listdir(img_dir)
    
    labels_df = load_labels(labels_file).columns[2:]
    
    predictions_labels = []
    x_test_filename = []
    
    model = keras.models.load_model(model_file)
   
    #continuar a partir de 5000
    idx=0
    while idx < 300:  #len(img_dir):
    
        #Carrega as imagens
        (X, file_names) = load_test_images(img_dir, jpg_list, labels_df, idx, idx + p_batch_size_load)
        
        #Predictions
        classes = model.predict(X, batch_size=p_batch_size_keras)
        
        for prediction in classes:
            labels = [labels_df[i] for i, value in enumerate(prediction) if value > 0.5]
            predictions_labels.append(labels)
        
        idx += p_batch_size_load
          
        tags_list = [None] * len(predictions_labels)
        for i, tags in enumerate(predictions_labels):
            tags_list[i] = ' '.join(map(str, tags))
    
        final_data = [[filename.split(".")[0], tags] for filename, tags in zip(file_names, tags_list)]
        
        final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
        
        if not os.path.isfile(sub_file):
            final_df.to_csv(sub_file,header ='column_names', index=False)
        else: # else it exists so append without writing the header
            final_df.to_csv(sub_file,mode = 'a',header=False, index=False)
        
        
#########################################################################################
    
#Executa o treinamento
#batch_training(p_batch_size_load=100, p_epochs=2, p_batch_size_keras=5)

#########################################################################################

