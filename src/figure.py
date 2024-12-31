import logging

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

        total_days_prediction = (
            self.output_prediction["future_pred"].index.max() - self.output_prediction["future_pred"].index.min()
        ).total_seconds() // 3600

        actual_legend = "Total system load (actual)<br>"
        pred_legend = (
            "Total system load (model prediction)<br>"
            f"Model training: MAE = {self.output_prediction['metrics_train']['mae']:.0f}, MAPE = {self.output_prediction['metrics_train']['mape']:.2f}<br>"
            f"Model prediction (24h): MAE = {self.output_prediction['metrics_future_one_day']['mae']:.0f}, MAPE = {self.output_prediction['metrics_future_one_day']['mape']:.2f}<br>"
            f"Model prediction ({total_days_prediction:.0f}h): MAE = {self.output_prediction['metrics_future']['mae']:.0f}, MAPE = {self.output_prediction['metrics_future']['mape']:.2f}<br>"
        )
        forecast_legend = (
            'Total system load (ENTSOE one-day forecast)<br>'
            f"Entsoe forecast (24h): MAE = {self.output_prediction['metrics_forecast_one_day']['mae']:.0f}, MAPE = {self.output_prediction['metrics_forecast_one_day']['mape']:.2f}<br>"
            f"Entsoe forecast ({total_days_prediction:.0f}h): MAE = {self.output_prediction['metrics_forecast']['mae']:.0f}, MAPE = {self.output_prediction['metrics_forecast']['mape']:.2f}"
        )

        self.fig.add_trace(go.Scatter(x=y_actual.index, y=y_actual, mode="lines+markers", name=actual_legend))
        self.fig.add_trace(go.Scatter(x=y_pred.index, y=y_pred, mode="lines+markers", name=pred_legend))
        self.fig.add_trace(go.Scatter(x=y_forecast.index, y=y_forecast, mode="lines+markers", name=forecast_legend))
        self.fig.update_layout(legend_valign="top")
        max_range = y_actual.index.max()
        min_range = max_range - pd.Timedelta(days=3)
        self.fig.update_xaxes(range=[min_range, max_range])
        min_range_y = 0.8 * y_actual.loc[min_range:max_range].min()
        max_range_y = 1.2 * y_actual.loc[min_range:max_range].max()
        self.fig.update_yaxes(range=[min_range_y, max_range_y])

        end_training = self.output_prediction["train_actual"].index.max()
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
            y=1.1,
        )

    def write_to_file(self) -> None:
        """Write Plotly plot to HTML file."""
        to_plot = {"fig": self.fig.to_html(full_html=False)}
        with open(c.path_output_html, "w", encoding="utf-8") as output_file:
            with open(c.path_template) as template_file:
                j2_template = Template(template_file.read())
                logging.info(f"Writing plot to {c.path_output_html}...")
                output_file.write(j2_template.render(to_plot))
