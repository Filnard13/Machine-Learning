# Project Work: Machine Learning for Economists

**University of Bologna — LM(EC)² Programme**  
**Last Updated:** 13/11/2025  

## Group Members
- Filippo Nardoni  
- Federica Carrieri  
- Luca Danelli  
- Anita Scambia  

## Project Title
**Forecasting Dynamics and Comovements in European Financial Markets through FARM**

## Description

This project performs an empirical analysis of **European stock market predictability** using:

- **Factor-augmented regression models (FARM Predict)**
- Benchmark models:
  - AR  
  - VAR  
  - PCR  
  - AR–PCR  
  - VAR–LASSO  

The main goal is to compare the forecasting performance of these models on a large panel of European stock returns and to study the comovements across markets.

---

## Folder Details

#DATA/

This folder contains all the data files used in the project
“Forecasting Dynamics and Comovements in European Financial Markets through FARM.”
Each file plays a specific role in the preprocessing, cleaning, and main forecasting pipeline.
	1.	EU_stocks_data_daily_2_new.csv
Main dataset of daily returns for all European stocks used in the analysis.
This file is read inside the main script code_project_final.m as the primary input for the forecasting models.
	2.	dates_daily_new.csv
Vector of trading dates corresponding to the observations in the main dataset.
Used for indexing, plotting with financial crisis shading, and aligning rolling windows.
	3.	Download_and_Data_Cleaning.R
R script used to download the original raw data and convert it into the cleaned format used in MATLAB.
Includes filtering, missing-value handling, and reformatting.



#TABLES/

Folder where all final tables (forecast evaluation, MSFE comparisons, etc.) are saved.


#CODE/
	1.	code_project_final.m
Main MATLAB script containing the empirical analysis and forecasting pipeline
(FARM Predict, AR, VAR, PCR, AR–PCR, VAR–LASSO, rolling windows, etc.).

## Repository Structure

```text
.
├── CODE/
│   └── code_project_final.m
├── DATA/
│   ├── EU_stocks_data_daily_2_new.csv
│   ├── dates_daily_new.csv
│   └── Download_and_Data_Cleaning.R
└── TABLES/



🔧 Important:
Before running, remember to change the initial file path in the script:
["insert/Your/Path"] so that it correctly points to your local folders DATA, TABLES, and CODE.

