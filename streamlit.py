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


# @st.cache_data
def get_data():
    dict_series = main.get_model_prediction()
    if dict_series is None:
        return None
    df_actual = treat_data(pd.concat([dict_series["train_actual"], dict_series["future_actual"]]), "actual")
    df_pred_train = treat_data(dict_series["train_pred"], "pred_train")
    df_pred_future = treat_data(dict_series["future_pred"], "pred_forecast")
    return pd.concat([df_actual, df_pred_train, df_pred_future], ignore_index=False), (
        dict_series["mape_train"],
        dict_series["mape_future"],
    )


output = get_data()
if output is None:
    st.text("No model trained!")
else:
    source, (mape_train, mape_future) = get_data()
    fig = px.line(
        source,
        x="time",
        y="demand",
        color="type",
        title="Energy demand in France",
        markers=True,
        labels={"time": "Time", "demand": "Energy demand (MW)", "type": ""},
    )

    newnames = {
        "actual": "Actual",
        "pred_train": f"Predicted training with MAPE of {np.round(mape_train,2)}",
        "pred_forecast": f"Predicted forecast with MAPE of {np.round(mape_future,2)}",
    }
    fig.for_each_trace(lambda t: t.update(name=newnames[t.name]))

    # print(source.loc[source["type"] == "pred_forecast", "time"].iloc[0])
    max_range = source.loc[source["type"] == "actual", "time"].iloc[-1]
    min_range = source.loc[source["type"] == "pred_forecast", "time"].iloc[0] - pd.Timedelta(days=3)
    fig.update_xaxes(range=[min_range, max_range])
    min_range_y = 0.8 * source.loc[source["time"] >= min_range, "demand"].min()
    max_range_y = 1.2 * source.loc[source["time"] >= min_range, "demand"].max()
    fig.update_yaxes(range=[min_range_y, max_range_y])
    st.plotly_chart(fig)
