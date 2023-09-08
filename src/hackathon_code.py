import pandas as pd
import numpy as np
import seaborn as sns
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from hugchat import hugchat
from hugchat.login import Login



open_df = pd.read_csv("AI_Hackathon_dataset.csv")
open_columns = list(open_df["indicator"].unique())

def data_wrangler(raw_df):
    final_features = {}
    features = raw_df["indicator"].unique()
    regions = raw_df["location"].unique()
    for region in regions:
        condition = raw_df["location"] == region
        specific_df = raw_df[condition]
        
        feature_list = []
        for feature in features:
            if feature in list(specific_df["indicator"]):
                feature_cond = specific_df["indicator"] == feature
                value_df = specific_df[feature_cond]
                value = float(value_df["value"])
            else:
                value = np.nan
            feature_list.append((feature, value))
        final_features[region] = feature_list
    return final_features
                
                

def df_constructor(data, cols):
    result = []
    for k,v in data.items():
        options = []
        options.append(k)
        for metrices in v:
            label, metric = metrices
            
            options.append(metric)
        result.append(options)
    
    master_df = pd.DataFrame(result, columns = ["location"] + cols)
    return master_df
        

clean_data = data_wrangler(open_df)
df = df_constructor(clean_data, open_columns)





df.isnull().sum()
df = df.set_index("location", drop = True)

insufficient_zones = ["Kano Central Sen. Dist.", "Kano North Sen. Dist.", "Kano South Sen. Dist.", 
                      "Lagos Central Sen. Dist.", "Lagos East Sen. Dist."]
df = df.drop(labels = insufficient_zones, axis = "index")

insufficient_cols = ['Availability of Skilled Birth Attendants excluding CHEWs', 'Percentage of Primary Health Facilities that Provide ANC Services', 'Percentage of Health Facilities that provide ANC Services', 
                     'Percentage of Health Facilities that provide Iron Supplements as part of Routine ANC services', 'Percentage of health facilities that provide folic acid supplements as part of routine ANC services', 
                     'Percentage of health facilities that provide Tetanus immunization as part of routine ANC services', 'Percentage of health facilities that provide Counselling about birth spacing or family planning as part of routine ANC services', 
                     'Percentage of health facilities offering STI services that had national guidelines for the diagnosis and treatment of STIs',
                     'Percentage of health facilities offering STI services that had syphilis test kits',  'Percentage of health facilities offering TB services that had the national guidelines for TB infection control', 
                     'Percentage of health facilities offering TB services that had first-line TB drugs',  'Percentage of health facilities that monitor for hypertensive disorder in pregnancy as part of routine ANC services',
                      'Percentage of health facilities that provide intermittent preventive treatment in pregnancy (IPTp) for malaria as part of routine ANC services', 
                      'Availability of Skilled Birth Attendants Including CHEWs', 'Percentage of health facilities offering TB services that have the national guideline for clinical management of TB and HIV/AIDS-related conditions in Nigeria', 
                      'Percentage of Primary Health Facilities with Implants or Intra-uterine contraceptive device', 'Percentage of Primary Health Facilities with Capacity to Provide Early Infant Diagnosis Services', 
                      'Percentage of Health Facilities that Conduct On-Site Early Infant Diagnosis Test', 'Percentage of Primary Health Facilities that have Malaria test Capacity', 'Percentage of health facilities offering STI services that had metronidazole tab/cap']

df = df.drop(labels = insufficient_cols, axis = "columns")

df.info()
data_descr = df.describe().transpose()



def fill_nulls(semi_clean_df):
    columns = list(semi_clean_df.columns)
    for column in columns:
        if semi_clean_df[column].isnull().sum() > 0:
            mean_val = semi_clean_df[column].mean()
            semi_clean_df[column].fillna(mean_val, inplace = True)
            
    return semi_clean_df



new_df = fill_nulls(df)


data_corr = new_df.corr()
#sns.heatmap(data_corr, annot = True)

mod2strong_corr_condition = data_corr >= 0.5
mod2str_corr = data_corr[mod2strong_corr_condition]
#plt.figure(figsize = (30, 22))
#sns.heatmap(mod2str_corr, annot = True, linewidths = 1, linecolor = "black")

negative_corr_condition = data_corr < 0 
negative_corr = data_corr[negative_corr_condition]
#plt.figure(figsize = (30, 22))
#sns.heatmap(negative_corr, annot = True, linewidths = 1, linecolor = "black")



class Athena():
    def __init__(self, data, plot = "without plot"):
        self.data = data
        self.scaler = MinMaxScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        self.pca_scale = PCA(2, random_state=101)
        self.df_pca = self.pca_scale.fit_transform(self.scaled_data)
        self.plot = plot
        
    def kmeans_clust(self):
        kmeans = KMeans(n_clusters= 3, random_state = 101, n_init = "auto")
        labels = kmeans.fit_predict(self.df_pca)
        self.pred_df_km = pd.DataFrame(self.df_pca, columns= ["dim1", "dim2"], index = self.data.index)
        df_km_copy = self.data.copy()
        self.pred_df_km["km_labels"] = labels
        df_km_copy["km_labels"] = labels
    
        if self.plot == "with plot":
            print(df_km_copy)
            sns.pairplot(self.pred_df_km, palette= "tab10", hue = "km_labels")
        return df_km_copy
    
    def auto_enc(self, epoch_num = 35):
        self.epoch_num = epoch_num
        
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
            sns.pairplot(self.pred_df_ae, palette= "tab10", hue = "Health Indices")
        return df_ae_copy
        
        
                
        
        
        
        




def athena_chat(df_r):
    if df_r.shape == (1, 29):  
        athena = Athena(df_r)
        df = athena.auto_enc()
        h_ind = df["Health Indices"]
        sign = Login("togunwataofeeq@gmail.com", "Ay0mide1*")
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
        print("Please only pass a single row of data. Ensure shape is (1, 30)")


athena_chat(x)

x = cc.iloc[18:19, :]


x["Health Indices"].values

test_sc = MinMaxScaler()
to = AthenaBeta(data = new_df, scaler = test_sc)    
cc = to.auto_enc()


x