import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from hugchat import hugchat
from hugchat.login import Login
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model




class Athena():
    # Initialize the Athena class with data and options for plotting.
    def __init__(self, data, plot = "without plot"):
        self.data = data
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        self.pca_scale = PCA(2, random_state=101)
        self.df_pca = self.pca_scale.fit_transform(self.scaled_data)
        self.plot = plot
        
    def kmeans_clust(self):
        # Perform K-Means clustering on the data and return the clustered DataFrame.
        try:
            kmeans = joblib.load("kmeans.joblib")
        except FileNotFoundError as e:
            kmeans = KMeans(n_clusters= 3, random_state = 101, n_init = "auto")
            joblib.dump(kmeans, "kmeans.joblib")
            
        labels = kmeans.fit_predict(self.df_pca)
        self.pred_df_km = pd.DataFrame(self.df_pca, columns= ["dim1", "dim2"], index = self.data.index)
        df_km_copy = self.data.copy()
        self.pred_df_km["km_labels"] = labels
        df_km_copy["km_labels"] = labels
    
        if self.plot == "with plot":
            print(df_km_copy)
            sns.pairplot(self.pred_df_km, palette= "tab10", hue = "km_labels").fig.suptitle(t = "Clustering Analysis for Data-Driven Similarity Profiling", y = 1.001)
        return df_km_copy
    
    def auto_enc(self):
        # Perform Autoencoder-based analysis and return the processed DataFrame.
        self.epoch_num = 35
        try:
            self.encoder = load_model("AutoEnc.h5").layers[0]
        except OSError as i:
            self.encoder = Sequential()
            self.encoder.add(Dense(21, "relu", input_shape = [29]))
            self.encoder.add(Dense(13, "relu"))
            self.encoder.add(Dense(5, "relu"))
            self.encoder.add(Dense(1, "relu"))
            
            self.decoder = Sequential()
            self.decoder.add(Dense(5, "relu", input_shape = [1]))
            self.decoder.add(Dense(13, "relu"))
            self.decoder.add(Dense(21, "relu"))
            self.decoder.add(Dense(29, "relu"))
    
            self.auto_encoder = Sequential([self.encoder, self.decoder])
            self.auto_encoder.compile(loss = "mse", optimizer = "adam")
            self.auto_encoder.fit(self.scaled_data, self.scaled_data, epochs = self.epoch_num)
            self.auto_encoder.save("AutoEnc.h5")
        
        self.linear_series = self.encoder.predict(self.scaled_data)
        
        self.top_q = np.quantile(self.linear_series, 0.75)
        self.mid_q = np.quantile(self.linear_series, 0.50)
        self.low_q = np.quantile(self.linear_series, 0.25)
        
        self.health_index = []
        for i in self.linear_series:
            if i <= self.low_q:
                self.health_index.append("Poor Health Index")
            elif i >= self.top_q:
                self.health_index.append("Good Health Index")
            else:
                self.health_index.append("Fair Health Index")
                
        self.pred_df_ae = pd.DataFrame(self.df_pca, columns= ["dim1", "dim2"], index = self.data.index)
        self.pred_df_ae["Health Indices"] = self.health_index
        df_ae_copy = self.data.copy()
        df_ae_copy["Health Indices"] = self.health_index
        
        if self.plot == "with plot":
            print(df_ae_copy)
            sns.pairplot(self.pred_df_ae, palette= "tab10", hue = "Health Indices").fig.suptitle(t = "Grouping of Health Indices", y = 1.001)
        return df_ae_copy
        
        



def athena_chat(df, username, password):
    # Initiate a chatbot conversation based on health metrics and recommendations.
    if df.shape == (1, 30):  
        h_ind = df["Health Indices"]
        sign = Login(username, password)
        cookies = sign.login()
        chatbot = hugchat.ChatBot(cookies = cookies.get_dict())
        id_chat = chatbot.new_conversation()
        chatbot.change_conversation(id_chat)
        prompt = f'''You are to refer to yourself as <Athena> and you are to respond as if <Athena> is speaking in prose. \
                You are a public health expert that summarizes health metrics and give recommendations to better health indices.\
                Your task is to examine just the health metrics provided in {df.to_dict()} and offer a summary, based on these metrics. ALWAYS ensure the summary MUST contain the "{h_ind}" value of the location. \
                        You are then to provide valuable and insightful recommendations to improve healthcare in the location, and achieve SDG3.\
                            You are to start both the summary and recommendation section with the word <Athena>. Ensure that your response isn't too technical or long, such that it can be understood and acted upon by non-experts'''
        print(chatbot.chat(prompt))
    else:
        print("Please only pass a single row of data. Ensure shape is (1, 29)")
        
        
        
        
        
        
if __name__ == "__main__":
    Athena()
    athena_chat()