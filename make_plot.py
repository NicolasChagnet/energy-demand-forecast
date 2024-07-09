import pandas as pd
import main
from src import config
import numpy as np
import plotly.express as px
from jinja2 import Template


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


def write_to_file(plot=None):
    if plot is None:
        to_plot = {"fig": "No model trained!"}
    else:
        to_plot = {"fig": plot.to_html(full_html=False)}
    with open(config.path_output_html, "w", encoding="utf-8") as output_file:
        with open(config.path_template) as template_file:
            j2_template = Template(template_file.read())
            output_file.write(j2_template.render(to_plot))


output = get_data()
if output is None:
    write_to_file(None)
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

    write_to_file(fig)
