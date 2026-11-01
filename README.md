# AgroSense AI
Python-only, software-only Streamlit web application.

## Run on your computer
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run `pip install -r requirements.txt`.
4. Run `streamlit run app.py`.

## Publish a public website
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Sign in at https://share.streamlit.io/ using GitHub.
3. Select **Create app**, choose your repository and branch, and set main file path to `app.py`.
4. Click Deploy. Streamlit will generate your public website URL.

The app starts with an empty database. Upload your own CSV or enter real readings. For ML training, upload a labelled dataset with numeric feature columns and a numeric target column. No synthetic/sample agricultural data is generated. Virtual pump control is only a software simulation.
