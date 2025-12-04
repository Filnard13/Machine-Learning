
%% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%
%                 Project Work: Machine Learning for Economists           %
%
%  University of Bologna — LM(EC)² Programme
%  Last Updated: 13/11/2025
%
%  Group Members:
%    • Filippo Nardoni
%    • Federica Carrieri
%    • Luca Danelli
%    • Anita Scambia
%
%  Description:
%    Empirical analysis of European stock market predictability using
%    factor-augmented regression models (FARM Predict) and related
%    benchmark approaches (AR, VAR, PCR, AR–PCR, VAR–LASSO).
%
% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ %%


This folder contains all the data files used in the project “Forecasting Dynamics and Comovements in European Financial Markets through FARM.”
Each file plays a specific role in the preprocessing, cleaning, and main forecasting pipeline.

FOLDER -> DATA
	1.	EU_stocks_data_daily_2_new.csv
Main dataset of daily returns for all European stocks used in the analysis.
This file is read inside the main script code_project_final.m as the primary input for the forecasting models.
	2.	dates_daily_new.csv
Vector of trading dates corresponding to the observations in the main dataset.
Used for indexing, plotting with financial crises shading, and aligning rolling windows.
	3.	Download_and_Data_Cleaning.R
R script used to download original raw data and convert it into the cleaned format used in MATLAB.
Includes filtering, missing-value handling, and reformatting.

FOLDER -> TABLES
         Folder for final savings of all tables.

FOLDER -> CODE
	1.	"code_project_final.m"
         Main Code with the empirical Analysis. Remember to change the initial file path [“insert/Your/Path”]

