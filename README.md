# Replication Data and Code 

This repository contains the data and Python scripts used to collect and analyze Steam reviews for this dissertation.

## Files in this Repository

1. **`100 AI & Non-AI List [Pre-Sentiment & Price].xlsx`**  
   The starting list of games used for the study.

2. **`build_dataset.py`**  
   FIRST SCRIPT. It connects to the Steam API, downloads recent English reviews for the games, runs the VADER sentiment analysis, and counts technical keywords. Saves results into a new file called `Master_Dataset_Analyzed.csv`.

3. **`Master_Dataset_Final.csv`**  
   Final dataset ready for analysis (contains the output from the first script plus the pricing data).

4. **`run_regressions_rbs.py`**  
   SECOND SCRIPT. It reads the final dataset, prepares the variables, and runs the three Ordinary Least Squares (OLS) regression models using robust standard errors (HC3). 

5. **`requirements.txt`**  
   A list of the Python packages needed to run the code.

---

## How to Run the Code

Please follow these steps to replicate study:

### Step 1: Install Python and Required Packages
Make sure you have Python installed on your computer and download repository files. Open your computer's terminal or relevant software to run script (I used Visual Studio Code), navigate to the folder containing these files, and install the required packages by typing:
`pip install -r requirements.txt`

### Step 2: Run the Data Collection Script
To extract the reviews and calculate the sentiment scores, run the first script:
`python build_dataset.py`

*Note: This script has built-in pauses to respect Steam's API rate limits. It will take some time to finish. When it is done, it will create a new CSV file with the results.*

### Step 3: Run the Regression Models
To run the statistical models and view the results, run the second script:
`python run_regressions_rbs.py`

When the script finishes, it will print the summaries of all three models to your screen. It will also create a new Excel file named `Regression_Results_Robust_Final.xlsx`.
