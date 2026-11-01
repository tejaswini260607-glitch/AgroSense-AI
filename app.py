import streamlit as st
import pandas as pd
import numpy as np
import sqlite3, os, joblib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px

st.set_page_config(page_title="AgroSense AI", page_icon="🌱", layout="wide")
ROOT=os.path.dirname(os.path.abspath(__file__))
DB=os.path.join(ROOT,"agrosense.db")
MODEL=os.path.join(ROOT,"agrosense_model.joblib")

def connect():
    c=sqlite3.connect(DB, check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS readings(
    id INTEGER PRIMARY KEY, recorded_at TEXT, crop TEXT, soil TEXT,
    soil_moisture REAL, temperature REAL, humidity REAL, rainfall REAL)""")
    c.execute("CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT)")
    c.commit()
    return c
con=connect()
def setting(k,d):
    r=con.execute("SELECT value FROM settings WHERE key=?",(k,)).fetchone()
    return r[0] if r else d
def save_setting(k,v):
    con.execute("INSERT OR REPLACE INTO settings VALUES(?,?)",(k,str(v))); con.commit()
def save_row(r):
    con.execute("INSERT INTO readings(recorded_at,crop,soil,soil_moisture,temperature,humidity,rainfall) VALUES(?,?,?,?,?,?,?)",
    (r["recorded_at"],r.get("crop",""),r.get("soil",""),r["soil_moisture"],r.get("temperature"),r.get("humidity"),r.get("rainfall")))
    con.commit()
def readings():
    return pd.read_sql_query("SELECT * FROM readings ORDER BY recorded_at",con)
def banner(title,sub):
    st.markdown(f"""<div style="background:linear-gradient(110deg,#123d2a,#26734a);padding:22px;border-radius:14px;color:white">
    <h1 style="color:white;margin:0">{title}</h1><p style="color:#e5f3e9;margin:5px 0 0">{sub}</p></div>""",unsafe_allow_html=True)

st.sidebar.title("🌱 AgroSense AI")
st.sidebar.caption("Predict Smart. Irrigate Smart. Grow Sustainable.")
page=st.sidebar.radio("Navigation",["Home Dashboard","Upload Agricultural Data","Manual Sensor Input","AI Prediction","Irrigation Management","Crop & Soil Settings","Analytics & Reports","About Project"])
st.sidebar.caption("Software-only decision support")

if page=="Home Dashboard":
    banner("AgroSense AI","Predict Smart. Irrigate Smart. Grow Sustainable.")
    st.write("Monitor your agricultural readings and make informed irrigation decisions using your own data.")
    df=readings()
    if df.empty: st.info("No agricultural data available. Please upload your dataset or enter sensor readings to continue.")
    else:
        x=df.iloc[-1]
        a,b,c,d=st.columns(4)
        a.metric("Soil moisture",f'{x.soil_moisture:.1f}%')
        b.metric("Temperature","—" if pd.isna(x.temperature) else f"{x.temperature:.1f} °C")
        c.metric("Humidity","—" if pd.isna(x.humidity) else f"{x.humidity:.1f}%")
        d.metric("Rainfall","—" if pd.isna(x.rainfall) else f"{x.rainfall:.1f} mm")
        st.subheader("Recent readings"); st.dataframe(df.tail(10),use_container_width=True,hide_index=True)
        st.plotly_chart(px.line(df,x="recorded_at",y="soil_moisture",markers=True,title="Soil moisture over time"),use_container_width=True)

elif page=="Upload Agricultural Data":
    banner("Upload Agricultural Data","Import your own CSV. No sample readings are generated.")
    st.write("Your CSV must include a numeric soil-moisture column. Other fields are optional.")
    f=st.file_uploader("Choose CSV file",type=["csv"])
    if f:
        try:
            raw=pd.read_csv(f); st.dataframe(raw.head(10),use_container_width=True)
            cols=list(raw.columns)
            moisture=st.selectbox("Soil moisture column (required)",cols)
            time=st.selectbox("Date/time column (optional)",["—"]+cols)
            crop=st.selectbox("Crop column (optional)",["—"]+cols)
            soil=st.selectbox("Soil type column (optional)",["—"]+cols)
            temp=st.selectbox("Temperature column (optional)",["—"]+cols)
            hum=st.selectbox("Humidity column (optional)",["—"]+cols)
            rain=st.selectbox("Rainfall column (optional)",["—"]+cols)
            if st.button("Import my data",type="primary"):
                out=pd.DataFrame()
                out["recorded_at"]=raw[time].astype(str) if time!="—" else datetime.now().isoformat(timespec="seconds")
                out["crop"]=raw[crop].astype(str) if crop!="—" else ""
                out["soil"]=raw[soil].astype(str) if soil!="—" else ""
                out["soil_moisture"]=pd.to_numeric(raw[moisture],errors="coerce")
                out["temperature"]=pd.to_numeric(raw[temp],errors="coerce") if temp!="—" else np.nan
                out["humidity"]=pd.to_numeric(raw[hum],errors="coerce") if hum!="—" else np.nan
                out["rainfall"]=pd.to_numeric(raw[rain],errors="coerce") if rain!="—" else np.nan
                out=out.dropna(subset=["soil_moisture"])
                if out.empty: st.error("No valid numeric moisture values found.")
                else:
                    for _,r in out.iterrows(): save_row(r.to_dict())
                    st.success(f"Imported {len(out)} real data rows.")
        except Exception as e: st.error(f"Could not read CSV: {e}")

elif page=="Manual Sensor Input":
    banner("Manual Sensor Input","Enter actual readings from your field or dataset.")
    with st.form("manual"):
        a,b=st.columns(2)
        crop=a.text_input("Crop name"); soil=b.selectbox("Soil type",["Not specified","Sandy","Clay","Loamy","Silty","Other"])
        moisture=a.number_input("Soil moisture (%)",0.0,100.0,0.0)
        temp=b.number_input("Temperature (°C)",value=0.0)
        hum=a.number_input("Humidity (%)",0.0,100.0,0.0)
        rain=b.number_input("Rainfall (mm)",0.0, value=0.0)
        ok=st.form_submit_button("Save reading",type="primary")
    if ok:
        save_row(dict(recorded_at=datetime.now().isoformat(timespec="seconds"),crop=crop,soil=soil,soil_moisture=moisture,temperature=temp,humidity=hum,rainfall=rain))
        st.success("Reading saved.")

elif page=="AI Prediction":
    banner("AI Prediction","Train only with your own labelled dataset.")
    st.warning("A valid future prediction requires real historical input features and a numeric target column containing measured future values. The app never invents target values.")
    f=st.file_uploader("Upload training CSV",type=["csv"],key="training")
    if f:
        try:
            df=pd.read_csv(f); st.dataframe(df.head(8),use_container_width=True)
            nums=df.select_dtypes(include=np.number).columns.tolist()
            if len(nums)<2: st.error("Need at least one numeric feature and one numeric target.")
            else:
                target=st.selectbox("Target column to predict",nums)
                features=st.multiselect("Input features",[c for c in nums if c!=target],default=[c for c in nums if c!=target])
                if st.button("Train AI model",type="primary"):
                    if not features: st.error("Choose at least one feature.")
                    else:
                        clean=df[features+[target]].apply(pd.to_numeric,errors="coerce").dropna()
                        if len(clean)<10: st.error(f"Only {len(clean)} complete rows. Please provide at least 10 real labelled rows.")
                        else:
                            Xtr,Xte,ytr,yte=train_test_split(clean[features],clean[target],test_size=.2,random_state=42)
                            model=RandomForestRegressor(n_estimators=150,random_state=42).fit(Xtr,ytr)
                            pred=model.predict(Xte); joblib.dump({"model":model,"features":features,"target":target},MODEL)
                            st.success("Model trained and saved.")
                            p,q=st.columns(2); p.metric("Mean absolute error",f"{mean_absolute_error(yte,pred):.3f}")
                            score=r2_score(yte,pred) if len(yte)>1 else np.nan
                            q.metric("R² score","Not available" if pd.isna(score) else f"{score:.3f}")
        except Exception as e: st.error(f"Training file error: {e}")
    if os.path.exists(MODEL):
        saved=joblib.load(MODEL); st.subheader("Make a prediction")
        vals={}; boxes=st.columns(min(3,len(saved["features"])))
        for i,name in enumerate(saved["features"]): vals[name]=boxes[i%len(boxes)].number_input(name,value=0.0,key="feature_"+name)
        if st.button("Predict",type="primary"):
            value=saved["model"].predict(pd.DataFrame([[vals[n] for n in saved["features"]]],columns=saved["features"]))[0]
            st.success(f'Predicted {saved["target"]}: {value:.2f}')

elif page=="Irrigation Management":
    banner("Irrigation Management","Recommendations only — no physical pump or hardware is controlled.")
    df=readings(); low=float(setting("min_moisture",30)); high=float(setting("max_moisture",70))
    st.write(f"Your configured moisture range: **{low:.1f}% to {high:.1f}%**")
    if df.empty: st.info("No agricultural data available. Please upload your dataset or enter sensor readings to continue.")
    else:
        value=float(df.iloc[-1].soil_moisture); st.metric("Latest soil moisture",f"{value:.1f}%")
        if value<low: st.warning("Soil moisture is below your minimum threshold. Consider irrigation after checking field conditions and rainfall.")
        elif value>high: st.info("Moisture is above your maximum threshold. Irrigation is not recommended from this reading.")
        else: st.success("Moisture is within your configured range. No irrigation recommendation is triggered.")
        st.caption("Virtual pump status is only an on-screen simulation.")
        status=st.selectbox("Virtual pump status",["OFF","ON"],index=0 if setting("pump","OFF")=="OFF" else 1)
        if st.button("Save virtual status"):
            save_setting("pump",status); st.success("Virtual status saved.")
        st.write("Saved status:",setting("pump","OFF"))

elif page=="Crop & Soil Settings":
    banner("Crop & Soil Settings","Set thresholds using your own crop guidance.")
    with st.form("settings"):
        crop=st.text_input("Crop name",setting("crop",""))
        soils=["Not specified","Sandy","Clay","Loamy","Silty","Other"]
        soil=st.selectbox("Soil type",soils,index=soils.index(setting("soil","Not specified")))
        low=st.number_input("Minimum moisture (%)",0.0,100.0,float(setting("min_moisture",30)))
        high=st.number_input("Maximum moisture (%)",0.0,100.0,float(setting("max_moisture",70)))
        ok=st.form_submit_button("Save settings",type="primary")
    if ok:
        if low>=high: st.error("Maximum must be greater than minimum.")
        else:
            for k,v in [("crop",crop),("soil",soil),("min_moisture",low),("max_moisture",high)]: save_setting(k,v)
            st.success("Settings saved.")

elif page=="Analytics & Reports":
    banner("Analytics & Reports","Visualize and export your own recorded data.")
    df=readings()
    if df.empty: st.info("No agricultural data available. Please upload your dataset or enter sensor readings to continue.")
    else:
        st.dataframe(df,use_container_width=True,hide_index=True)
        choices=[c for c in ["soil_moisture","temperature","humidity","rainfall"] if df[c].notna().any()]
        metric=st.selectbox("Measurement",choices)
        st.plotly_chart(px.line(df,x="recorded_at",y=metric,markers=True),use_container_width=True)
        st.download_button("Download CSV",df.to_csv(index=False).encode(), "agrosense_readings.csv","text/csv")

else:
    banner("About AgroSense AI","MRITHON 3.0 | Edge AI – Predictive Agriculture with Intelligent Irrigation")
    st.markdown("""
    **AgroSense AI** is a software-only agriculture decision-support prototype.

    **Technology:** Python, Streamlit, Pandas, NumPy, Scikit-learn, Plotly, SQLite and Joblib.

    **Important:** No sensors, Arduino, ESP32, Raspberry Pi, relays or physical pumps are connected.
    No sample or synthetic agricultural data is created. ML training requires your own real,
    labelled dataset. Irrigation suggestions are informational and should be checked against
    actual field conditions and local agronomic guidance.
    """)
