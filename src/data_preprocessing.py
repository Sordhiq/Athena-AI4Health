import pandas as pd
import numpy as np



def data_wrangler(raw_df):
    # Extracts unique indicator and location values, creates indicator-value pairs by location.
    # Returns a dictionary of features for each location.
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
    # Constructs a DataFrame with location and selected indicator columns.
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



def preprocessing(unclean_df):   
    # Cleans the DataFrame by setting an index and removing specific zones and indicators.
    df = unclean_df.set_index("location", drop = True)
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
    return df


def fill_nulls(semi_clean_df):
    # Fills missing values in the DataFrame with their respective mean values.
    columns = list(semi_clean_df.columns)
    for column in columns:
        if semi_clean_df[column].isnull().sum() > 0:
            mean_val = semi_clean_df[column].mean()
            semi_clean_df[column].fillna(mean_val, inplace = True)
            
    return semi_clean_df


if __name__ == "__main__":
    data_wrangler()
    df_constructor() 
    preprocessing()
    fill_nulls()
