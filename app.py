"""
A simple app demonstrating how to dynamically render tab content containing
dcc.Graph components to ensure graphs get sized correctly. We also show how
dcc.Store can be used to cache the results of an expensive graph generation
process so that switching tabs is fast.
"""
import pickle

import time

import dash
from dash.dependencies import Input, Output
import dash_table
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from lib.functions_dash import reviews_by_score, reviews_by_date, get_noise_words, get_common_words, main_ngrams
from lib.functions_model import get_main_words, main_words, get_examples_tables

###==============================================CODIGO===========================================================
df = pd.read_csv('Data/brazilian-ecommerce/olist_order_reviews_dataset.csv')
df_or = pd.read_csv('Data/brazilian-ecommerce/olist_order_reviews_clean_dataset.csv')
df_or = df_or[df_or.review_comment_message_1.notna()]
df.review_creation_date = pd.to_datetime(df.review_creation_date)
df.review_answer_timestamp = pd.to_datetime(df.review_answer_timestamp)

##PAGINA 1
#----Variables panel 1
num_reviews = len(df)
num_com_reviews = df.review_comment_message.count()
perc_pos = len(df[(df.review_score == 4) | (df.review_score == 5)])/len(df)
perc_neu = len(df[(df.review_score == 3)])/len(df)
perc_neq = len(df[(df.review_score == 1) | (df.review_score == 2)])/len(df)

#----Graficas panel 2
bar_reviews_by_score = reviews_by_score(df)
line_reviews_by_date = reviews_by_date(df)

#----Fraficas panel 3
noise_words = get_noise_words(df_or)
print('-------------------')
df_3grama_good = get_common_words(df_or, noise_words, 1)
df_3grama_bad = get_common_words(df_or, noise_words, 0)

bar_3gram_good = main_ngrams(df_3grama_good)
bar_3gram_bad = main_ngrams(df_3grama_bad, 0)

##PAGINA 2
#---Modelos
loaded_model = pickle.load(open('models/linear_model.sav', 'rb'))
vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))

df_main_words = get_main_words(vectorizer,loaded_model)
bar_words_good=main_words(df_main_words)
bar_words_bad=main_words(df_main_words, sentiment = 0)

df_pos_examples = get_examples_tables(df_or[df_or.review_sentiment == 1].iloc[0:100,:],
                                      vectorizer,loaded_model)
df_neq_examples = get_examples_tables(df_or[df_or.review_sentiment == 0].iloc[0:100,:],
                                      vectorizer,loaded_model)

###============================================APLICACION=========================================================

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.config['suppress_callback_exceptions'] = True

app.layout = dbc.Container(
    [
        dcc.Store(id="store"),
        html.H1("Olist Analisis de Sentimiento"),
        html.Hr(),
#         dbc.Button(
#             "Regenerate graphs",
#             color="primary",
#             block=True,
#             id="button",
#             className="mb-3",
#         ),
        dbc.Tabs(
            [
                dbc.Tab(label="An??lisis de sentimiento", tab_id="page-1"),
                dbc.Tab(label="Modelo ML", tab_id="page-2"),
            ],
            id="tabs",
            active_tab="page-1",
        ),
        html.Div(id="tab-content", className="p-4"),
    ],
)


@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab == "page-1":
        page_1 = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H4('{:,}'.format(num_reviews)),
                                                    html.P("No. Revisiones"),
                                                ])
                                            )
                                        ),
                                        dbc.Col(
                                            dbc.Card(
                                                dbc.CardBody([
                                                    html.H4('{:,}'.format(num_com_reviews)),
                                                    html.P("No. Comentarios")]),
                                            )
                                        ),
                                    ]
                                ),
                            ),
                            width=6
                        ),
                        dbc.Col(
                            html.Div(
                                dbc.Row([
                                    dbc.Col(dbc.Card(dbc.CardBody([html.H4('{:.2f}%'.format(perc_pos*100)),
                                                                  html.P("Positivos")]))),
                                    dbc.Col(dbc.Card(dbc.CardBody([html.H4('{:.2f}%'.format(perc_neu*100)),
                                                                  html.P("Neutros")]))),
                                    dbc.Col(dbc.Card(dbc.CardBody([html.H4('{:.2f}%'.format(perc_neq*100)),
                                                                  html.P("Negativos")]))),
                                ])
                            ),
                            width=6
                        )
                    ],
                    align="center",
                    justify="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Revisiones por puntaje"),
                                    dcc.Graph(figure = bar_reviews_by_score, id = 'bar_rev_sco')
                                ]
                            ),
                            width=6
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Revisiones por mes"),
                                    dcc.Graph(figure = line_reviews_by_date, id = 'lin_rev_dat')
                                ]
                            ),
                            width=6
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Trigramas comentarios positivos"),
                                    dcc.Graph(figure = bar_3gram_good, id = 'bar_good_ngr')
                                ]
                            ),
                            width=6
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Trigramas comentarios negativos"),
                                    dcc.Graph(figure = bar_3gram_bad, id = 'bar_bad_ngr')
                                ]
                            ),
                            width=6
                        ),
                    ]
                ),
            ]
        )
        return page_1
    elif active_tab == "page-2":
        page_2 = html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                dbc.Input(id="input-on-submit", placeholder="Ingrese el texto a clasificar...", type="text"),
                            ),
                            width=6
                        ),
                        dbc.Col(
                            dbc.Button('Submit', id='submit-val', color="primary", style={'width': '100%'}),
                            width=2
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Sentimiento Asociado", className="card-text"),
                                    html.P(id="sa-output", className="card-title",
                                          style={'font-size':'25px'}),
                                ])
                            ),
                            width=2
                        ),
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody([
                                    html.P("Probabilidad de clasificaci??n", className="card-text"),
                                    html.P(id="pc-output", className="card-title",
                                          style={'font-size':'25px'}),
                                ])
                            ),
                            width=2
                        )
                    ],
                    align="center",
                    justify="center",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Palabras principales en textos positivos"),
                                    dcc.Graph(figure = bar_words_good, id = 'bar_good_mw')
                                ]
                            ),
                            #width=6
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Ejemplos comentarios positivos"),
                                    dash_table.DataTable(id = 'table_top_cities',
                                                        columns=[{"name": i, "id": i} for i in df_pos_examples.columns],
                                                        data=df_pos_examples.to_dict('records'),
                                                        page_size=10,
                                                        style_cell={'whiteSpace': 'normal','height': 'auto'},
                                                        style_table={
                                                            'maxHeight': '50ex',
                                                            'overflowY': 'scroll',
                                                            'width': '100%',
                                                            'minWidth': '100%',
                                                        },
                                                        )
                                ]
                            ),
                            md = 8#width=6
                        ),
                    ]
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Palabras principales en textos negativos"),
                                    dcc.Graph(figure = bar_words_bad, id = 'bar_bad_mw')
                                ]
                            ),
                            #width=6
                        ),
                        dbc.Col(
                            html.Div(
                                [
                                    html.H3("Ejemplos comentarios negativos"),
                                    dash_table.DataTable(id = 'table_top_cities',
                                                        style_table={
                                                            'maxHeight': '50ex',
                                                            'overflowY': 'scroll',
                                                            'width': '100%',
                                                            'minWidth': '100%',
                                                        }, 
                                                        style_cell={'whiteSpace': 'normal','height': 'auto'},
                                                        columns=[{"name": i, "id": i} for i in df_neq_examples.columns],
                                                        data=df_neq_examples.to_dict('records'),
                                                        page_size=10,
                                                        )
                                ]
                            ),
                            md = 8
                        ),
                    ]
                )
            ]
        )
        return page_2

@app.callback(
    [
        Output('sa-output', 'children'),
        Output('pc-output', 'children'),
    ],
    [Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('input-on-submit', 'value')])
def update_output(n_clicks, value):
    if value is None:
        return "None", "0"
    else:
        Text = vectorizer.transform([value])
        
        sentiment = loaded_model.predict(Text)
        prob_sent = loaded_model.predict_proba(Text)
        
        if sentiment[0]: sentiment = 'Positivo'
        else: sentiment = 'Negativo'
            
        prob_sent = '{:.2f}'.format(prob_sent[0][1])
    return str(sentiment), prob_sent

if __name__ == "__main__":
    app.run_server(debug=True, port=8050, dev_tools_ui=False,dev_tools_props_check=False)