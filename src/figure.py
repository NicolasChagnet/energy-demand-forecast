import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template

import src.config as c

logger = logging.getLogger(c.logger_name)


class PredictionFigure:
    def __init__(self, output_prediction: dict):
        self.output_prediction = output_prediction
        self.fig = go.Figure()

    def make_plot(self):
        y_actual = pd.concat([self.output_prediction["train_actual"], self.output_prediction["future_actual"]])
        y_pred = pd.concat([self.output_prediction["train_pred"], self.output_prediction["future_pred"]])
        y_forecast = self.output_prediction["future_forecast"]
        y_last_week = y_actual.shift(7 * 24)

        total_hours_prediction = (
            self.output_prediction["future_pred"].index.max()
            - self.output_prediction["future_pred"].index.min()
            + pd.Timedelta(hours=1)
        ).total_seconds() // 3600
        end_training = self.output_prediction["train_actual"].index.max()

        actual_legend = "Total system load (actual)<br>"
        pred_legend = (
            "Total system load (model prediction)<br>"
            f"Model training: MAE = {self.output_prediction['metrics_train']['mae']:.0f}, MAPE = {self.output_prediction['metrics_train']['mape']:.2f}<br>"
            f"Model prediction (24h): MAE = {self.output_prediction['metrics_future_one_day']['mae']:.0f}, MAPE = {self.output_prediction['metrics_future_one_day']['mape']:.2f}<br>"
            f"Model prediction ({total_hours_prediction:.0f}h): MAE = {self.output_prediction['metrics_future']['mae']:.0f}, MAPE = {self.output_prediction['metrics_future']['mape']:.2f}<br>"
        )
        forecast_legend = (
            'Total system load (ENTSOE one-day forecast)<br>'
            f"Entsoe forecast (24h): MAE = {self.output_prediction['metrics_forecast_one_day']['mae']:.0f}, MAPE = {self.output_prediction['metrics_forecast_one_day']['mape']:.2f}<br>"
            f"Entsoe forecast ({total_hours_prediction:.0f}h): MAE = {self.output_prediction['metrics_forecast']['mae']:.0f}, MAPE = {self.output_prediction['metrics_forecast']['mape']:.2f}<br>"
        )
        last_week_legend = "Total system load (actual, last week)"

        self.fig.add_trace(go.Scatter(x=y_actual.index, y=y_actual, mode="lines+markers", name=actual_legend))
        self.fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode="lines+markers", name=pred_legend))
        self.fig.add_trace(go.Scatter(x=y_forecast.index, y=y_forecast, mode="lines+markers", name=forecast_legend))
        self.fig.add_trace(
            go.Scatter(
                x=y_last_week.index,
                y=y_last_week,
                mode="lines+markers",
                line=dict(dash="dash"),
                name=last_week_legend,
                # opacity=0.6,
            )
        )
        self.fig.update_layout(
            autosize=False,
            width=1400,
            height=800,
            legend=dict(
                # itemwidth=100,
                yref="paper",
                orientation="h",  # show entries horizontally
                xanchor="center",  # use center of legend as anchor
                x=0.5,
                y=-0.15,
            ),
            xaxis=dict(automargin="height"),
        )
        self.fig.update_layout(legend_valign="top")
        max_range = y_actual.index.max()
        min_range = end_training - pd.Timedelta(days=1)
        self.fig.update_xaxes(range=[min_range, max_range])
        min_range_y = 0.8 * y_actual.loc[min_range:max_range].min()
        max_range_y = 1.2 * y_actual.loc[min_range:max_range].max()
        self.fig.update_yaxes(range=[min_range_y, max_range_y])

        self.fig.add_vline(
            x=end_training,
            line_width=2,
            line_color="black",
            line_dash="dash",
        )
        self.fig.add_annotation(
            # xanchor="left",
            x=end_training,
            text="End of training period",
            showarrow=False,
            yref="paper",
            y=1.05,
        )

    def write_to_file(self) -> None:
        """Write Plotly plot to HTML file."""
        to_plot = {"fig": self.fig.to_html(full_html=False)}
        with open(c.path_output_html, "w", encoding="utf-8") as output_file:
            with open(c.path_template) as template_file:
                j2_template = Template(template_file.read())
                logging.info(f"Writing plot to {c.path_output_html}...")
                output_file.write(j2_template.render(to_plot))

    def save_to_file(self, path_to_file: Path | str, **kwargs) -> None:
        """Save figure to file."""
        self.fig.write_image(path_to_file, **kwargs)
