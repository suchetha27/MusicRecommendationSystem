from flask import Flask
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
from math import pi, trunc
from flask import Flask,render_template,redirect,request
import pickle
from sklearn.model_selection import train_test_split
import time
import joblib
import requests
import pickle
from models import Recommenders
import seaborn as sns
import tkinter as tk
from tkinter import *
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
from flask import Flask, redirect, url_for, render_template, request
import numpy as np
from csv import reader
import warnings
from fer import FER
import pandas as pd
import numpy as np
import random
import csv
from flask import jsonify
import matplotlib.pyplot as plt 
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__,template_folder='templates',static_folder='static')

#triplets_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\10000.txt'
#songs_metadata_file = 'C:\\Users\\Chandana C L\\Documents\\MAJOR PROJECT\\song_data.csv'
data_file = 'https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv'

#song_df_1 = pandas.read_table(triplets_file, header=None)
#song_df_1.columns = ['user_id', 'song_id', 'listen_count']

#song_df_2 = pandas.read_csv(songs_metadata_file)
song_df = pd.read_csv(data_file, header=1, sep=",")
song_df.columns = ['id','song','singer','language','genre','movie/album','listen_count','user id']
#song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

song_df.head()

length = len(song_df)

song_df = song_df.head(10000)
song_df['song'] = song_df['song'].map(str)+"-"+song_df['singer']
song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()

song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped = song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])


@app.route('/', methods=['GET','POST'])
def predict():

    data_file = 'https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv'

    #song_df_1 = pandas.read_table(triplets_file, header=None)
    #song_df_1.columns = ['user_id', 'song_id', 'listen_count']

    #song_df_2 = pandas.read_csv(songs_metadata_file)
    song_df = pd.read_csv(data_file, header=1, sep=",")
    song_df.columns = ['id','song','singer','language','genre','movie/album','listen_count','user id']
    #song_df = pandas.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on="song_id", how="left")

    song_df.head()
    length = len(song_df)

    song_df = song_df.head(10000)
    song_df['song'] = song_df['song'].map(str)+"-"+song_df['singer']
    song_grouped = song_df.groupby(['song']).agg({'listen_count': 'count'}).reset_index()
    grouped_sum = song_grouped['listen_count'].sum()
    song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
    song_grouped = song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
    

    if request.method== "POST":
        int_features = request.form["song"]
        songs = pd.read_csv("https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs.csv", header=1, sep=",")
        songs.columns = ['id','title','singer','language','genre','movie/album','listen count','user id']

        song_user = songs.groupby('user id')['id'].count()
        song_ten_id = song_user[song_user > 16].index.to_list()
        df_song_id_more_ten = songs[songs['user id'].isin(song_ten_id)].reset_index(drop=True)

        df_songs_features = df_song_id_more_ten.pivot(index='id', columns='user id', values='listen count').fillna(0)
        mat_songs_features = csr_matrix(df_songs_features.values)

        class Recommender():
            def __init__(self, metric, algorithm, k, data, decode_id_song):
                self.metric = metric
                self.algorithm = algorithm
                self.k = k
                self.data = data
                self.decode_id_song = decode_id_song
                #self.model = self._recommender().fit(data)
                self.model = NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1).fit(data)
      

            def _recommender(self):
                 return NearestNeighbors(metric=self.metric, algorithm=self.algorithm, n_neighbors=self.k, n_jobs=-1)

            def make_recommendation(self, new_song, n_recommendations):
                recommended = self._recommend(new_song=new_song, n_recommendations=n_recommendations)
                print("... Done")
                return recommended 

            def _recommend(self, new_song, n_recommendations):
                # Get the id of the recommended songs
                recommendations = []
                recommendation_ids = self.get_recommendation(new_song=new_song, n_recommendations=n_recommendations)
                # return the name of the song using a mapping dictionary
                recommendations_map = self._map_indeces_to_song_title(recommendation_ids)
                # Translate this recommendations into the ranking of song titles recommended
                for i, (idx, dist) in enumerate(recommendation_ids):
                    recommendations.append(recommendations_map[idx])
                return recommendations

            def get_recommendation(self, new_song, n_recommendations):
                recom_song_id = self._fuzzy_matching(song=new_song)
                # Return the n neighbors for the song id
                distances, indices = self.model.kneighbors(self.data[recom_song_id], 
                                                           n_neighbors=n_recommendations+1)
                return sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), 
                              key=lambda x: x[1])[:0:-1]

            def _map_indeces_to_song_title(self, recommendation_ids):
                # get reverse mapper
                return {song_id: song_title for song_title, song_id in self.decode_id_song.items()}


            def _fuzzy_matching(self, song):
                match_tuple = []
                # get match
                for title, idx in self.decode_id_song.items():
                    ratio = fuzz.ratio(title.lower(), song.lower())
                    if ratio >= 40:
                        match_tuple.append((title, idx, ratio))
            
                match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    
                return match_tuple[0][1]

        df_unique_songs = songs.drop_duplicates(['id']).reset_index(drop=True)[['id', 'title']]
        decode_id_song = {
            song: i for i, song in 
            enumerate(list(df_unique_songs.set_index('id').loc[df_songs_features.index].title))
            }

        model = Recommender(metric='cosine', algorithm='brute', k=20, data=mat_songs_features, decode_id_song=decode_id_song)
        new_recommendations = model.make_recommendation(new_song=int_features, n_recommendations=10)

    #     return render_template("main.html", data=new_recommendations, songs=song_grouped['song'].head(15))
    # else:
    #     return render_template("main.html", songs=song_grouped['song'].head(15))

        data=""
        song = request.form["song"]
        df=pd.read_csv("https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs%20-%20majorsongs.csv")
        #print(df.columns)
        def get_title_from_index(index):
            return df[df.index == index]["title"].values[0]

        def get_index_from_title(title):
            return df[df.title == title]["id"].values[0]
        ##Step 2: Select Features
        features=['singer','genre','movie/album']
        for feature in features:
            df[feature]=df[feature].fillna('')
        def combine_features(row):
            try:
                return row['singer']+" "+row['genre']+" "+row['movie/album']
            except:
                print("ERROR"+row)

        df["combined_features"]=df.apply(combine_features,axis=1)

        cv=CountVectorizer()
        cm=cv.fit_transform(df["combined_features"])
        ##Step 5: Compute the Cosine Similarity based on the count_matrix

        ss=cosine_similarity(cm)
        #music_user_likes =input("Enter song you like ")
        music_user_likes=song
        ## Step 6: Get index of this music from its title
        try:
            music_index=get_index_from_title(music_user_likes)
            similar_music=list(enumerate(ss[music_index]))
            sorted_similar_music=sorted(similar_music,key=lambda x:x[1],reverse=True)
            ## Step 7: Get a list of similar music in descending order of similarity score


            ## Step 8: Print titles of first 20 music 
            song_list=[]
            i=0
            ###### helper functions. Use them when needed #######

            for music in sorted_similar_music:
                if(get_title_from_index(music[0])==music_user_likes):
                    continue
                else:
                    song_list.append(get_title_from_index(music[0]))
                    i=i+1
                    if i>19:
                        break
            return render_template("main.html",data=song_list,songs=song_grouped['song'].head(15),sn=new_recommendations)
        except:
            err="No such song"
            return render_template("main.html",err=err, songs=song_grouped['song'].head(15))
    else:
        return render_template("main.html", songs=song_grouped['song'].head(15))


@app.route("/emotion", methods=['GET', 'POST'])
def emotion():
    if request.method == "POST":
        n = request.form["nm"]
        ind=int(n)
        face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
        classifier =load_model('./Model.h5')

        class_labels = ['angry','happy','neutral','sad','surprise']

        cap = cv2.VideoCapture(0)

        result=True

        while result:
            # Grab a single frame of video
            ret, frame = cap.read()
            labels = []
            
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h,x:x+w]
                roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                

                if np.sum([roi_gray])!=0:
                    roi = roi_gray.astype('float')/255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi,axis=0)

                # make a prediction on the ROI, then lookup the class
                #predict probability of each emotion
                    preds = classifier.predict(roi)[0]          # predict the labels of the data values on the basis of the trained model.
                    print("\nProbability of emotions\n")
                    print(class_labels)
                    print(preds)
                    label=class_labels[preds.argmax()]
                    print("\nprediction max = ",preds.argmax())         #identify the index of maximum probability of emotion
                    print("\nemotion of the image = ",label)
                    label_position = (x,y)
                    cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                    cv2.imwrite(label+".jpg",frame)
                    cap.release()
                else:
                    cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                print("\n\n")
                result=False
            cv2.imshow('Emotion Detector',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        reader=pd.read_csv("https://s3.ap-geo.objectstorage.softlayer.net/musicdataset/majorsongs%20-%20majorsongs.csv",header=0)
        song_list=[]
        # print(reader)
        genreData = reader.loc[:, 'genre' ]
        titleData=reader.loc[:, 'title']
        emo=""
        for i in range(len(genreData)):
            genre=str(genreData[i])
            genre=genre.lower()
            if(label=="neutral"):
                if genre.count("classical") or genre.count("folk"):
                    song_list.append(titleData[i])

            elif(label=="surprise"):
                if genre.count("dance"):
                    song_list.append(titleData[i])

            elif(label=="happy"):
                if genre.count("romantic") or genre.count("funny"):
                    song_list.append(titleData[i])

            elif(label=="sad"):
                if genre.count("sad") or genre.count("patho"):
                    song_list.append(titleData[i])
            
            elif(label=="angry"):
                if genre.count("rap"):
                    song_list.append(titleData[i])

        if(ind<=len(song_list)):
            return render_template('emotion.html',emo=label,data=song_list[:ind],songs=song_grouped['song'].head(15))
        else:
            msg=" ** Number of songs you want ("+n+") is greater than number of songs suitable ("+str(len(song_list))+") for "+label+" emotion. "
            return render_template('emotion.html',msg=msg,emo=label,data=song_list[:ind],songs=song_grouped['song'].head(15))
    else:
        return render_template('emotion.html',songs=song_grouped['song'].head(15))


if __name__ == "__main__":
    app.run(debug=True)




# sad=346(367=sad , 12=patho)       91.3% correct
# angry=4/4(rap=4)                  correct
# happy=750(romantic=744, funny=6)   correct
# surprise=1283 (dance=1300)          98.7% correct
# neutral=95(folk=7, classical=88)  correct
