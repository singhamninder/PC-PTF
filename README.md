# Pseudo-continuous Pedotransfer Functions (PC-PTF) for SWRC Estimation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pc-ptf.streamlit.app/)

This repository contains the source code for a Streamlit web application designed to predict the Soil Water Retention Curve (SWRC) using machine learning models known as Pseudo-continuous Pedotransfer Functions (PC-PTFs).

## Overview

Pedotransfer Functions (PTFs) are predictive models that estimate soil hydraulic properties from more easily measured soil data, like texture and bulk density. This application implements a set of *pseudo-continuous* PTFs, which use an ensemble of 100 neural networks to predict the volumetric water content (VWC) across a continuous range of soil water potentials (pF).

The application allows users to upload their own soil data and receive detailed SWRC predictions, including uncertainty estimates. For certain input combinations, it also provides a comparison with the well-established Rosetta3 model.

## Key Features

-   **Easy Data Upload**: Upload your soil data via a simple CSV file.
-   **Dynamic Model Selection**: The app automatically detects which predictive models are compatible with your provided data columns.
-   **Ensemble Predictions**: Leverages a bagging approach with 100 pre-trained Keras models for robust and reliable predictions.
-   **Uncertainty Quantification**: Calculates and displays the mean and standard deviation of the ensemble predictions.
-   **Rosetta3 Comparison**: For models using soil texture or texture + bulk density, it runs the Rosetta3 model as a baseline comparison.
-   **Interactive Visualization**: Plot the predicted SWRC for any random soil sample from your dataset.
-   **Downloadable Results**: Export the complete prediction results to a CSV file for further analysis.

## Citation

If you use this application, please cite the original paper:

>Singh, A., Verdi, A., 2024. Estimating the soil water retention curve by the HYPROP-WP4C system, HYPROP-based PCNN-PTF and inverse modeling using HYDRUS-1D. *Journal of Hydrology* 639, 131657. [https://doi.org/10.1016/j.jhydrol.2024.131657](https://doi.org/10.1016/j.jhydrol.2024.131657)

## How to Use the Web App

1.  **Prepare Your Data**: Create a CSV file with your soil sample data.
2.  **Column Naming**: Ensure your file contains a `soil#` column to identify unique samples and at least the three soil texture columns. The required column names are:
    -   `soil#`: A unique identifier for each soil sample.
    -   `clay`: Clay content (%)
    -   `silt`: Silt content (%)
    -   `sand`: Sand content (%)
    -   `BD`: Bulk Density (g/cm³)
    -   `omc`: Organic Matter Content (%)
3.  **Visit the App**: Navigate to the PC-PTF Streamlit App.
4.  **Upload**: Use the sidebar to upload your CSV file. The app will validate the columns.
5.  **Select a Model**: Choose one of the available models from the dropdown menu. The options are determined by the columns in your file.
6.  **Analyze**: View the resulting data table, plot the SWRC for a random sample, and download the results using the provided buttons.
