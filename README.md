# Airbnb Price Classification - NYC
The main task we will be working on is supervised learning, classification. We will define target as  price: ‘low’, 'medium’, ‘high’ using quantile-based bins or possibly cluster-based bins, derived using  unsupervised learning, to explore underlying trends. 

## Team Members
### Part 1 -
- Asja Bašović (A) - Basic EDA, Data Cleaning, Feature Engineering
- Esma Kaderić (B) - Train-Test Split, Missing Data, Feature Selection  
- Amina Hrustić (C) - Full EDA, Clustering
### Part 2
- Asja Bašović  - 
- Esma Kaderić  - 
- Amina Hrustić - 
...

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-team/airbnb-project.git
cd airbnb-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Place the Airbnb NYC dataset in:
```
data/raw/airbnb_nyc.csv
```

### 4. Run the notebook
```bash
jupyter notebook airbnb_classification.ipynb
```

## Project Structure
```
airbnb-ml-project/
│           
├── data/
│   ├── raw/
│   │   └── ML_dfc.csv          # Dataset (not in git)
│   └── processed/ 
│ 
├── models/
├── notebooks/
│   ├── 01_data_preprocessing.ipynb #Asjas part is pasted here
│   ├── 02_feature_engineering_clustering.ipynb
│   └── 03_modeling_feature_selection.ipynb
├── README.md
├── report/
│   └── Report.pdf  
└──  requirements.txt               # Python dependencies
```

## Workflow
- Each section has a primary owner (marked A, B, or C)
- All members contribute to each section
- Use relative paths only: `'../data/raw/airbnb_nyc.csv'`
- Commit frequently with clear messages