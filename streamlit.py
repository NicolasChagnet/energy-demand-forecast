import pandas as pd
import main
import numpy as np
import plotly.express as px
import streamlit as st


def treat_data(y, name):
    y.index.name = "time"
    y.name = "demand"
    df = y.to_frame().reset_index()
    df["type"] = name
    return df


@st.cache_data
def get_data():
    (y, y_pred), mape = main.get_model_prediction()
    df1 = treat_data(y, "Actual")
    df2 = treat_data(y_pred, "Predicted")
    return pd.concat([df1, df2], ignore_index=False), mape


source, mape = get_data()
fig = px.line(
    source,
    x="time",
    y="demand",
    color="type",
    title="Energy demand in France",
    markers=True,
    labels={"time": "Time", "demand": "Energy demand (MW)", "type": ""},
)

newnames = {"Actual": "Actual", "Predicted": f"Predicted with MAPE of {np.round(mape,2)}"}
fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))

st.plotly_chart(fig)

# st.altair_chart((chart).interactive(), use_container_width=True)
