import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
import os
import pickle
import random


def create_Xtest(data):
    """Function to generate the data to be used as input to the model"""
    pF = pd.Series(np.arange(-1, 6, 0.05), name='pF')
    df = pd.concat([data.drop_duplicates()]*len(pF), ignore_index=True)
    test = pd.concat([df,pF], axis=1)
    return test

def wettodry(vwc):
    vwc = [0 if i < 0 else i for i in vwc]
    w2d = [vwc[0]]
    for idx in range(1, len(vwc)):
        if vwc[idx] < w2d[idx-1]:
            w2d.append(vwc[idx])
        else:
            w2d.append(w2d[idx-1])
    return w2d

def drytowet(vwc):
    vwc = [0 if i < 0 else i for i in vwc]
    rslt_reverse = vwc[::-1]
    d2w = [rslt_reverse[0]]
    for idx in range(1, len(vwc)):
        if rslt_reverse[idx] > d2w[idx-1]:
            d2w.append(rslt_reverse[idx])
        else:
            d2w.append(d2w[idx-1])
    return d2w[::-1]

@st.cache
def bag_predict(models, X_test):
    bag_pred = pd.DataFrame()
    for ann in models:
        ann_all = tf.keras.models.load_model(ann)
        y_pred = ann_all.predict(X_test)
        w2d = wettodry(y_pred.ravel().tolist())
        d2w = drytowet(y_pred.ravel().tolist())
        pred_list = [(g + h) / 2 for g, h in zip(w2d, d2w)]
        bag_pred[ann[ann.find('ann'):-3]] = pd.Series(pred_list)

    mean_vwc = bag_pred.mean(axis=1)
    std_vwc = bag_pred.std(axis=1)
    return mean_vwc, std_vwc

def plot_results(results_df):
    rand_soil = random.choice(results_df['soil#'].unique())
    soil_test = results_df[results_df['soil#']==rand_soil]
        
    sns.set(font_scale=1.5, style="ticks")
    fig, ax = plt.subplots(figsize=(6, 6))
    # ax.plot(test_F1['pF'], test_F1['VWC'], 'o', alpha=0.6, label = 'VWC')
    ax.plot(soil_test['pF'], soil_test['mean_vwc'], '-', label = 'PCPTF')
    ax.fill_between(soil_test['pF'], soil_test['mean_vwc']+soil_test['std_vwc'],
                    soil_test['mean_vwc']-soil_test['std_vwc'], alpha=0.3)
    ax.set_xlim(0,5)
    ax.legend(prop={'size': 14})
    ax.grid(True,linestyle='--')
    # ax.vlines(x = 4.2, ymin = 0, ymax = 0.6,
    #        colors = 'tab:red',
    #        label = 'PWP')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('pF',size = 18, weight = 'bold')
    ax.set_ylabel('VWC'r' [$cm^3 cm^{-3}$]',size = 18, weight = 'bold')
    ax.set_title("soil# = "+ str(rand_soil))
    return fig


st.title('Pseudo-continuous Pedotransfer Functions for estimating Soil Water Retention Curve (SWRC)')
with st.sidebar:
    st.subheader('Upload your CSV file')
    uploaded_file = st.file_uploader("Make sure coloumns are named - soil#, clay, silt, sand, bd, and omc", type=["csv"],
                help='File should atleast have columns - soil#, clay, silt, and sand')

    st.markdown('Clay [%],  Silt [%], Sand [%], ' 
                'Bulk Density'r' $[cm^3 cm^{-3}]$, ' 'and Organic Matter Content [%]')

st.subheader('Dataset')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**Glimpse of your dataset**')
    st.write(df)

    if set(['clay', 'silt', 'sand', 'bd', 'omc']).issubset(df.columns):
        st.success('All required columns are present')
        choices = (['model1', 'model2', 'model3', 'model4'])

    elif set(['clay', 'silt', 'sand', 'bd']).issubset(df.columns):
        st.success('clay, silt, sand, and bulk density(bd) columns are present')
        choices = (['model2', 'model3'])
    
    elif set(['clay', 'silt', 'sand', 'omc']).issubset(df.columns):
        st.success('clay, silt, sand, and soil organic matter(omc) columns are present')
        choices = (['model2', 'model4'])

    elif set(['clay', 'silt', 'sand']).issubset(df.columns):
        st.success('clay, silt, and sand columns are present')
        choices = (['model2'])
    
    else:
        st.error('Please make sure required columns are present and are named correctly')
    st.write(f"Data has {df.shape[0]} rows and {df.shape[1]} columns")

## MODEL FITTING
    st.header('Fit the Model')
    model = st. selectbox('Select model', choices)

    if model == 'model1':
        st.info(f'{model} uses soil texture(SSC), bulk density(bd), and soil organic matter (omc) as inputs')
        models = ['model1/ann_'+ str(i) + '.h5' for i in range(100)]
        ## generate the test dataset for the model
        colList = ['soil#', 'clay', 'silt', 'sand', 'bd', 'omc']
        df_group = df[colList].groupby(['soil#'])
        test_df = df_group.apply(create_Xtest).reset_index(drop=True)
        #load scaler and fit the models
        scaler = pickle.load(open('ann1_stdscaler.pkl', 'rb'))
       
        estimated_vwc = {}
        for soil, soil_test in test_df.groupby('soil#'):
            X_test = scaler.transform(soil_test.iloc[:,1:])
            mean_vwc, std_vwc = bag_predict(models, X_test)
            estimated_vwc[soil] = pd.DataFrame({'mean_vwc':mean_vwc,'std_vwc':std_vwc})
            vwc_ = pd.concat(estimated_vwc.values(), keys=estimated_vwc.keys()) \
                                .reset_index(level=0) \
                                .rename({'level_0':'soil#'},axis=1).reset_index(drop=True)
            results_df = pd.concat([test_df, vwc_.iloc[:,1:]],axis=1)

        st.markdown('**Results for your data**')
        st.write(results_df)

    elif model == 'model2':
        st.info(f'{model} uses soil texture(SSC) as input')
        models = ['model2/ann_'+ str(i) + '.h5' for i in range(100)]
        ## generate the test dataset for the model
        colList = ['soil#', 'clay', 'silt', 'sand']
        df_group = df[colList].groupby(['soil#'])
        test_df = df_group.apply(create_Xtest).reset_index(drop=True)
        #load scaler and fit the models
        scaler = pickle.load(open('ann2_stdscaler.pkl', 'rb'))
       
        estimated_vwc = {}
        for soil, soil_test in test_df.groupby('soil#'):
            X_test = scaler.transform(soil_test.iloc[:,1:])
            mean_vwc, std_vwc = bag_predict(models, X_test)
            estimated_vwc[soil] = pd.DataFrame({'mean_vwc':mean_vwc,'std_vwc':std_vwc})
            vwc_ = pd.concat(estimated_vwc.values(), keys=estimated_vwc.keys()) \
                                .reset_index(level=0) \
                                .rename({'level_0':'soil#'},axis=1).reset_index(drop=True)
            results_df = pd.concat([test_df, vwc_.iloc[:,1:]],axis=1)

        st.markdown('**Results for your data**')
        st.write(results_df)       

    elif model == 'model3':
        st.info(f'{model} uses soil texture(SSC), and bulk density(bd) as inputs')
        models = ['model3/ann_'+ str(i) + '.h5' for i in range(100)]
        ## generate the test dataset for the model
        colList = ['soil#', 'clay', 'silt', 'sand', 'bd']
        df_group = df[colList].groupby(['soil#'])
        test_df = df_group.apply(create_Xtest).reset_index(drop=True)
        #load scaler and fit the models
        scaler = pickle.load(open('ann3_stdscaler.pkl', 'rb'))
    
        estimated_vwc = {}
        for soil, soil_test in test_df.groupby('soil#'):
            X_test = scaler.transform(soil_test.iloc[:,1:])
            mean_vwc, std_vwc = bag_predict(models, X_test)
            estimated_vwc[soil] = pd.DataFrame({'mean_vwc':mean_vwc,'std_vwc':std_vwc})
            vwc_ = pd.concat(estimated_vwc.values(), keys=estimated_vwc.keys()) \
                                .reset_index(level=0) \
                                .rename({'level_0':'soil#'},axis=1).reset_index(drop=True)
            results_df = pd.concat([test_df, vwc_.iloc[:,1:]],axis=1)

        st.markdown('**Results for your data**')
        st.write(results_df)

    elif model == 'model4':
        st.info(f'{model} uses soil texture(SSC), and bulk density(bd) as inputs')
        models = ['model4/ann_'+ str(i) + '.h5' for i in range(100)]
        ## generate the test dataset for the model
        colList = ['soil#', 'clay', 'silt', 'sand', 'omc']
        df_group = df[colList].groupby(['soil#'])
        test_df = df_group.apply(create_Xtest).reset_index(drop=True)
        #load scaler and fit the models
        scaler = pickle.load(open('ann4_stdscaler.pkl', 'rb'))
    
        estimated_vwc = {}
        for soil, soil_test in test_df.groupby('soil#'):
            X_test = scaler.transform(soil_test.iloc[:,1:])
            mean_vwc, std_vwc = bag_predict(models, X_test)
            estimated_vwc[soil] = pd.DataFrame({'mean_vwc':mean_vwc,'std_vwc':std_vwc})
            vwc_ = pd.concat(estimated_vwc.values(), keys=estimated_vwc.keys()) \
                                .reset_index(level=0) \
                                .rename({'level_0':'soil#'},axis=1).reset_index(drop=True)
            results_df = pd.concat([test_df, vwc_.iloc[:,1:]],axis=1)

        st.markdown('**Results for your data**')
        st.write(results_df) 


    st.subheader('Plot Results')
    if st.button('Plot random soil', help='SWRC for a random soil from your dataset will be created at each click of this button'):
        fig = plot_results(results_df)
        st.pyplot(fig)
        st.markdown("***pF***: the logarithmic transformation of soil tension in ***cm*** of water")

    st.subheader('Download Results')
    csv = results_df.to_csv(index=False)
    st.download_button("Download Results", csv, "PCPTF_results.csv",
                "text/csv", key='download-csv')      

else:
    st.info('Awaiting for CSV file to be uploaded.')


