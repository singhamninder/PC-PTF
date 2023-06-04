import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn as sns
from rosetta import Rosetta
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

def van_genuchten(h, parms):
    theta_r = parms[0]
    theta_s = parms[1]
    alpha = 10**parms[2]
    neta = 10**parms[3]
    m = 1-1/neta

    numt = (theta_s- theta_r)
    Se=(1+abs(alpha*h)**neta)**(m)
    vwc = theta_r + numt/Se
    return vwc

def plot_results(results_df):
    """
    Plot the results of soil tests.

    Args:
        results_df (DataFrame): DataFrame containing soil test results.

    Returns:
        fig (Figure): Matplotlib Figure object containing the plot.
    """
    rand_soil = random.choice(results_df['soil#'].unique())   
    sns.set(font_scale=1.5, style="ticks")
    fig, ax = plt.subplots(figsize=(6, 6))

    if not rosetta_vwc_df.empty:
        soil_test = results_df[results_df['soil#'] == rand_soil]
        ax.plot(soil_test['pF'], soil_test['mean_vwc'], '-', label='PCPTF')
        ax.fill_between(soil_test['pF'], soil_test['mean_vwc'] + soil_test['std_vwc'],
                        soil_test['mean_vwc'] - soil_test['std_vwc'], alpha=0.3)
        rosetta_soil = rosetta_vwc_df[rosetta_vwc_df['soil#'] == rand_soil]
        ax.plot(rosetta_soil['pF'], rosetta_soil['rosetta'], '-', label='Rosetta3')
        ax.fill_between(rosetta_soil['pF'], rosetta_soil['std'], rosetta_soil['_std'], alpha=0.3)
    else:
        soil_test = results_df[results_df['soil#'] == rand_soil]
        ax.plot(soil_test['pF'], soil_test['mean_vwc'], '-', label='PCPTF')
        ax.fill_between(soil_test['pF'], soil_test['mean_vwc'] + soil_test['std_vwc'],
                        soil_test['mean_vwc'] - soil_test['std_vwc'], alpha=0.3)

    ax.set_xlim(0, 5)
    ax.legend(prop={'size': 14})
    ax.grid(True, linestyle='--')
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('pF', size=18, weight='bold')
    ax.set_ylabel('VWC [$cm^3 cm^{-3}$]', size=18, weight='bold')
    ax.set_title("soil# = " + str(rand_soil))
    
    return fig