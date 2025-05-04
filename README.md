# Cherry Harvest Dashboard

A Streamlit dashboard for analyzing and predicting cherry harvest data.

## Setup

1. Install Miniconda or Anaconda if you haven't already:
   - [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
   - [Anaconda](https://www.anaconda.com/products/distribution)

2. Create the conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate cherry
```

## Running the App

1. Make sure you're in the project directory and the conda environment is activated:
```bash
conda activate cherry
```

2. Run the Streamlit app:
```bash
streamlit run app/Home.py
```

The app will open in your default web browser.

## Environment Details

The conda environment includes:
- Python 3.10
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Seaborn
- PyTorch
- Custom src package