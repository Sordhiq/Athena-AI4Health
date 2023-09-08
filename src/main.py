from data_preprocessing import data_wrangler, df_constructor, preprocessing, fill_nulls
from train import Athena, athena_chat
import pandas as pd


# Load the raw data
open_df = pd.read_csv("AI_Hackathon_dataset.csv")

# Extract unique indicators
open_columns = list(open_df["indicator"].unique())

# Perform data preprocessing
clean_data = data_wrangler(open_df)
df_const = df_constructor(clean_data, open_columns)
df_prep = preprocessing(df_const)


# Exploratory data analysis
#df_prep.isnull().sum()
#df_prep.info()
#data_descr = df_prep.describe().transpose()


# Fill missing values
new_df = fill_nulls(df_prep)


# Calculate data correlation
data_corr = new_df.corr()
#sns.heatmap(data_corr, annot = True)


# Identify moderate to strong correlations
mod2strong_corr_condition = data_corr >= 0.5
mod2str_corr = data_corr[mod2strong_corr_condition]
#plt.figure(figsize = (30, 22))
#sns.heatmap(mod2str_corr, annot = True, linewidths = 1, linecolor = "black")


# Identify negative correlations
negative_corr_condition = data_corr < 0 
negative_corr = data_corr[negative_corr_condition]
#plt.figure(figsize = (30, 22))
#sns.heatmap(negative_corr, annot = True, linewidths = 1, linecolor = "black")


# Initialize Athena for further analysis
athena = Athena(new_df, plot = "with plot")

# Categorize health indices using autoencoder
health_index_pred = athena.kmeans_clust()
df_health_index = pd.DataFrame(health_index_pred.loc["Bayelsa"]).transpose()


# Initiate a chatbot conversation
athena_chat()


