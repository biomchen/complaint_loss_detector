#!usr/bin/env python3

import base64
import io
import json
import os
import pandas as pd
import dash
from dash import Input, Output, dcc, html, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import plotly.express as px
import joblib

from utils import *

BS = "https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css"
app = dash.Dash(external_stylesheets=[BS])

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    'color':'#5cbdb9',
    'font-size':36,
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

FUNCTION_STYLE =  {
    "margin-left": "8rem",
    "margin-right": "8rem",
    "margin-bottom": "20px"
}

server = app.server

app.layout = html.Div(
    [
        html.H1(
            'Complaint Loss Detector',
            style=CONTENT_STYLE
        ),
        dbc.Row(
            [
                dbc.Col(
                    html.P(
                        "A NLP and ML powered tool for monetary loss reduction",
                        style={
                            'color':'#2d545e',
                            'font-size':16,
                            "margin-left":"11rem",
                            "margin-bottom": "10px",
                        },
                    ),
                    width=8
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            "Info", 
                            id='click-info', 
                            n_clicks=0,
                            style=FUNCTION_STYLE,
                            outline=True,
                            color="info"),
                        dbc.Offcanvas(
                            [
                                html.P(
                                    "This project focus on detecting the severity of "
                                    "the consumers' complaints filed at the "
                                    "Consumer Financial Protection Burea against "
                                    "the finanical insitutions at the United States."
                                ),
                                html.Br(),
                                html.H5('Data'),
                                html.P("In total, 3.1M consumers' complaints against 6,594 financial institutions."),
                                html.P("1.68M complaints against other financial institutions, with ~7% causing monetary loss."),
                                html.Br(),
                                html.H5('Tech'),
                                html.P(
                                    "Python, pandas, scikit-Learn, seaborn"
                                    "nltk, spaCy, geopy, folium, wordcloud, "
                                    "feature engineering, NaiveBayes, "
                                    "Logistic Regression, Xgboost, "
                                    "Dash, AWS EC2."
                                ),
                            ],
                            id='offcanvas',
                            title='CLD Project Information',
                            is_open=False
                        ),
                    ],
                    width=4
                ),
            ],
        ),
        html.Div(
            [
                dbc.Alert(
                    "I. Single Complaint Analyzer", 
                    color="primary",
                    style={"margin-left":"20px"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Textarea(
                                size="lg", 
                                id='single-complaint-val',
                                placeholder="Copy and paste the single complaint here ...",
                                style={
                                    "font-size":12,
                                    "width":"580px",
                                    "height":"100px",
                                    "lineHeight":"20px",
                                    "borderWidth":"1px",
                                    "borderStyle":"dashed",
                                    "boarderRadius":"5px",
                                    "margin-left":"40px",
                                    "margin-bottom": "10px"
                                },
                            ),
                            width=10
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    'Submit', 
                                    id='single-complaint-submit', 
                                    outline=True,
                                    color="primary", 
                                    n_clicks=0,
                                    style={
                                        "margin-left":"10px",
                                        "height":"40px",
                                        "margin-bottom":"10px"
                                    },
                                ),
                            ],
                            width=2
                        ),
                    ],
                ),
                dbc.Alert(
                    'Prediction',
                    id='prediction',
                    color="warning",
                    style={
                        "margin-left":"20px",
                    }
                ),
            ],style=FUNCTION_STYLE,
        ),           
        html.Div(
            [
                dbc.Alert(
                    "II. Complaint Batch Processor", 
                    color="primary",
                    style={"margin-left":"20px"}
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Upload(
                                id='upload-batch-data',
                                children=html.Div([
                                    'drag and drop or ', 
                                    html.A('select a csv file')
                                ]), 
                                style={
                                    "width":"455px",
                                    "height":"40px",
                                    "lineHeight":"40px",
                                    "borderWidth":"1px",
                                    "borderStyle":"dashed",
                                    "boarderRadius":"5px",
                                    "textAlign":"center",
                                    "margin-left":"40px",
                                    "margin-bottom": "10px"
                                },
                                multiple=False,
                            ),
                            width=10
                        ),
                        dbc.Col(
                            dbc.Button(
                                'Submit', 
                                id='complaint-batch-val', 
                                outline=True,
                                color="primary", 
                                n_clicks=0,
                                style={
                                    "margin-left":"10px",
                                    "height":"40px",
                                    "margin-bottom": "10px"
                                },
                            ),
                            width=2
                        ),
                    ],
                ),
                html.Div(id='batch-data'),
            ], style=FUNCTION_STYLE,
        ),
        html.Div(
            [
                dbc.Alert(
                    'Data Geographic Distribution',
                    color="secondary",
                    style={"margin-left":"20px"}
                ),
                dbc.Row(
                    dbc.Spinner(
                        [dcc.Graph(id='geo-plot')], 
                        color='light',
                    ),
                    style={"margin-left":"10px"}
                ),
                html.Br(),
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Alert(
                                'Results', 
                                color="secondary",
                                style={"margin-left":"20px"}
                            ),
                            width=10,
                        ),
                        dbc.Col(
                            [
                                dbc.Button(
                                    'Download', 
                                    id='results-download', 
                                    outline=True,
                                    color="primary", 
                                    n_clicks=0,
                                    style={
                                        "height":"40px",
                                        "margin-bottom":"10px"
                                    },
                                ),
                                dcc.Download(
                                    id='predict-results-download'
                                ),
                            ], 
                            width=2,
                        ),
                    ],
                ),
            ], style=FUNCTION_STYLE,
        ),
    ],
)


file_path = os.getcwd() + '/model.joblib'
model = joblib.load(open(file_path, 'rb'))
state_coords = json.load(open('state_coords.json', 'rb'))

response_dict = {
    '0':'Low probability of monetary loss',
    '1': 'Likely causing monetary loss'
}

def parse_data(contents, filename):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if "csv" in filename:
            # for csv or text file
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        elif "xls" in filename:
            # for excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div(["There was an error processing this file."])
    return df

@app.callback(
    Output("offcanvas", "is_open"),
    Input("click-info", "n_clicks"),
    [State("offcanvas", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('prediction', 'children'),
    Input('single-complaint-submit', 'n_clicks'),
    State('single-complaint-val', 'value'),
)
def predict_severity_single(n_clicks, text):
    if not text:
        raise PreventUpdate
    print(text)
    prediction = model.predict([text])
    ans = response_dict[str(prediction[0])]
    print(ans)
    return ans

@app.callback(
    Output("geo-plot", "figure"),
    [
        Input('complaint-batch-val', 'n_clicks'),
        Input('upload-batch-data', 'contents'),
        Input('upload-batch-data', 'filename')
    ],
)
def geo_plot(n_clicks, contents, filename):
    if n_clicks == 0:
        raise PreventUpdate
    df = parse_data(contents, filename)
    counts = df['State'].value_counts()
    states = pd.DataFrame(counts).reset_index()
    states.columns = ['state', 'count']
    lats, longs = [], []
    for state in states['state']:
        lats.append(state_coords[state][0])
        longs.append(state_coords[state][1])
    states['lat'] = lats
    states['long'] = longs
    print(states)

    figure = px.scatter_mapbox(
        states,
        lat='lat', lon='long',
        size='count',
        hover_name="state",
        color_discrete_sequence=["#e3a700"],
        center={'lat':38.8283, 'lon': -98.5795},
        zoom=2.8, 
        width=700, 
        height=400)

    figure.update_layout(
        mapbox_style="open-street-map",
        plot_bgcolor='#232323',
        paper_bgcolor='#232323',
        font_color='#7FDBFF',
        margin=dict(l=0, r=0, t=0, b=0))
    return figure

@app.callback(
    Output("predict-results-download", "data"),
    [
        Input('results-download', 'n_clicks'),
        Input('upload-batch-data', 'contents'),
        Input('upload-batch-data', 'filename')
    ],
)
def predict_severity_batch(n_clicks, contents, filename):
    if n_clicks == 0:
        raise PreventUpdate
    df = parse_data(contents, filename)    
    texts = df['Consumer complaint narrative']
    predictions = model.predict(texts)
    predictions = pd.Series(predictions)
    print(predictions)
    ans = pd.concat(
        [
            df[['Complaint ID', 'Consumer complaint narrative']], 
            predictions
        ], 
        axis=1
    )
    ans.columns = [
        'complaint_id', 'consumer_complaint narrative', 'prediction'
    ]
    return dcc.send_data_frame(
        ans.to_csv, "predicted_results.csv")


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=3000)
