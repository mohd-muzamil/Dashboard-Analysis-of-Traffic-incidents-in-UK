from dash import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import calendar
from readData import *
from timeSeriesPrediction import *
from dbScan import *


# Defining some function which will be used later in the code
def plot_map(df, color='N'):
    px.set_mapbox_access_token(
        'pk.eyJ1IjoibW9oZDA4MTIiLCJhIjoiY2toemozc2Z1MGx1ejJ4cGRmenVuMGR5biJ9.cu7UmDyn1L6BeO730TRS3w')

    if color == 'Y':
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude", color="Cluster",
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
    else:
        fig = px.scatter_mapbox(df, lat="Latitude", lon="Longitude",
                                color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

        fig.update_layout(height=800)
    return fig


def plot_heatmap(df):
    center = dict(lat=df['Latitude'].mean(), lon=df['Longitude'].mean())
    fig = px.density_mapbox(df, lat='Latitude', lon='Longitude',  # z='Magnitude',
                            radius=10,
                            center=center, zoom=10,
                            mapbox_style="open-street-map"
                            )
    # fig.update_layout(height=800)
    return fig


file = open("hotspots.html", "w+")
file.close()

accident_data = read_preprocessed_accidents_data()
combined_data = readAccidentVehiclesData()

color_scale = [[0.0, 'rgb(247,252,253)'],
               [0.1111111111111111, 'rgb(224,236,244)'],
               [0.2222222222222222, 'rgb(224,236,244)'],
               [0.3333333333333333, 'rgb(191,211,230)'],
               [0.4444444444444444, 'rgb(158,188,218)'],
               [0.5555555555555556, 'rgb(140,150,198)'],
               [0.6666666666666666, 'rgb(140,107,177)'],
               [0.7777777777777778, 'rgb(136,65,157)'],
               [0.8888888888888888, 'rgb(129,15,124)'],
               [0.9999999999999999, 'rgb(69,117,180)'],
               [1.0, 'rgb(77,0,75)']]

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

all_options = {
    'Time Range': ['month', 'hour', 'Day_of_Week', 'week_in_month'],
    'Data Attributes': ['Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 'Road_Type', 'Speed_limit',
                        'Light_Conditions', 'Weather_Conditions', 'Road_Surface_Conditions', 'Urban_or_Rural_Area']
}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MATERIA])

app.layout = html.Div([
    dbc.Card(
        dbc.CardBody([
            html.H3(children='Visualization and Analysis of Traffic Incidents in UK',
                    style={'textAlign': 'center', 'color': 'red'}),
            html.Br(),

            dcc.Tabs([
                dcc.Tab(label='Exploratory Data Analysis', children=[
                    # Tab1 Code here
                    html.Br(),
                    dbc.Row([
                        # Tab1 First Row First Column Here
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label('Select Year', id='slider_value'),
                                dcc.RangeSlider(id="slider", min=np.sort(accident_data.year.unique())[0],
                                                max=np.sort(accident_data.year.unique())[-1],
                                                value=[np.sort(accident_data.year.unique())[4],
                                                       np.sort(accident_data.year.unique())[5]],
                                                marks={x: str(x) for x in range(np.sort(accident_data.year.unique())[0],
                                                                                (np.sort(accident_data.year.unique())[
                                                                                    -1]) + 1, 1)}),

                            ]),
                        ], width=9, className="p-3 mb-2 text-dark"),

                        # Tab1 First Row Second Column Here
                        dbc.Col([
                            html.Div(id='accident_count', className="p-3 mb-2 text-danger")], width=3)
                    ]),

                    dbc.Row([

                        dbc.Col([
                            # Tab1 Second Row First Column Here
                            html.Div(children=html.Strong('Monthly Accident counts over Selected Time'),
                                     style={'textAlign': 'center'}),
                            dbc.FormGroup([
                                html.Strong(dbc.Label("Choose Graph")),
                                dcc.Dropdown(id="dropdown", value=1, clearable=False,
                                             options=[dict(label='Bar', value=1), dict(label='Line', value=2)]),
                            ]),

                            dbc.Col(dcc.Graph(id='example_graph', style={'height': '50vh'})),
                        ], width=6),

                        # Tab1 Second Row Second Column Here
                        dbc.Col([
                            html.Div(children=html.Strong("Choose Grouping column for Analysis"),
                                     style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='column_type_dropdown', clearable=False,
                                options=[{'label': k, 'value': k} for k in all_options.keys()],
                                value='Time Range'
                            ),
                            dcc.Dropdown(id='column_values_dropdown', clearable=False, ),
                            html.Br(),
                            html.Div(dcc.Graph(id='combined_agg_graph')),
                        ], width=6)

                    ]),
                    html.Br(), html.Br(),

                    dbc.Row([
                        # Tab1 Third Row First Column Here
                        dbc.Col([
                            html.Div(children=html.Strong("Choose columns for Treemap Analysis"),
                                     style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='columns_dropdown_treemap', clearable=False,
                                options=[
                                    {'label': 'Police_Force', 'value': 'Police_Force'},
                                    {'label': 'Accident_Severity', 'value': 'Accident_Severity'},
                                    {'label': 'Number_of_Vehicles', 'value': 'Number_of_Vehicles'},
                                    {'label': 'Day_of_Week', 'value': 'Day_of_Week'},
                                    {'label': 'Road_Type', 'value': 'Road_Type'},
                                    {'label': 'Speed_limit', 'value': 'Speed_limit'},
                                    {'label': 'Light_Conditions', 'value': 'Light_Conditions'},
                                    {'label': 'Weather_Conditions', 'value': 'Weather_Conditions'},
                                    {'label': 'Road_Surface_Conditions', 'value': 'Road_Surface_Conditions'},
                                    {'label': 'Urban_or_Rural_Area', 'value': 'Urban_or_Rural_Area'},
                                    {'label': 'month', 'value': 'month'},
                                    {'label': 'hour', 'value': 'hour'},
                                    {'label': 'day', 'value': 'day'},
                                    {'label': 'week_in_month', 'value': 'week_in_month'},
                                    {'label': 'Vehicle_Type', 'value': 'Vehicle_Type'},
                                    {'label': 'Sex_of_Driver', 'value': 'Sex_of_Driver'},
                                    {'label': 'Age_Band_of_Driver', 'value': 'Age_Band_of_Driver'},
                                    {'label': 'Engine_Capacity_(CC)', 'value': 'Engine_Capacity_(CC)'},
                                    {'label': 'Age_of_Vehicle', 'value': 'Age_of_Vehicle'}
                                ],
                                value=['Accident_Severity'], multi=True,
                            ),
                            html.Div(dcc.Graph(id='treemap')),

                        ], width=7),

                        # Tab1 Third Row Second Column Here
                        html.Br(), html.Br(),
                        dbc.Col([
                            html.Div(children=html.Strong("Correlation Between Columns"),
                                     style={'textAlign': 'center'}),
                            dcc.Graph(id='correlation'),
                        ], width=5)])

                ]),

                dcc.Tab(label='Time Series Prediction', children=[
                    # Tab2 Code here
                    html.Br(),
                    dbc.Row([
                        # Tab2 First Row First Column Here
                        dbc.Col([
                            html.Div(children=html.Strong('Time Series prediction using ARIMA model'),
                                     style={'textAlign': 'center'}),
                            html.Strong(dbc.Label("Choose Time Scale")),
                            html.Div([
                                dcc.RadioItems(id='time_aggregate',
                                               options=[
                                                   {'label': 'Yearly', 'value': 'Y'},
                                                   {'label': 'Monthly', 'value': 'M'},
                                                   {'label': 'Weekly', 'value': 'W'}
                                               ],
                                               value='M'
                                               )
                            ]),

                            html.Strong(dbc.Label("Choose Grouping column for Analysis")),
                            html.Div([
                                dcc.Dropdown(id="grouping_col", value='Accident_Severity', clearable=False,
                                             options=[
                                                 {'label': 'Accident_Severity', 'value': 'Accident_Severity'},
                                                 {'label': 'Number_of_Vehicles', 'value': 'Number_of_Vehicles'},
                                                 {'label': 'Number_of_Casualties', 'value': 'Number_of_Casualties'},
                                                 {'label': 'Road_Type', 'value': 'Road_Type'},
                                                 {'label': 'Speed_limit', 'value': 'Speed_limit'},
                                                 {'label': 'Light_Conditions', 'value': 'Light_Conditions'},
                                                 {'label': 'Weather_Conditions', 'value': 'Weather_Conditions'},
                                                 {'label': 'Road_Surface_Conditions',
                                                  'value': 'Road_Surface_Conditions'},
                                                 {'label': 'Urban_or_Rural_Area', 'value': 'Urban_or_Rural_Area'},
                                             ])
                            ]),
                            html.Br(),
                            html.Div(id='RMSE_Value'),
                            html.Div(dcc.Graph(id='time_series_figure')),

                        ], width=12),

                        # Tab2 First Row Second Column Here
                        dbc.Col([
                        ], width=0)])

                ]),
                dcc.Tab(label='Clustering', children=[
                    # Tab3 Code here
                    html.Br(),
                    dbc.Row([
                        # Tab3 First Row First Column Here
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label('Select Year', id='slider_value_1'),

                                dcc.Slider(id="slider_single_year_1", min=np.sort(combined_data.year.unique())[0],
                                           max=np.sort(combined_data.year.unique())[-1], step=1000,
                                           value=np.sort(combined_data.year.unique())[5],
                                           marks={x: str(x) for x in range(np.sort(combined_data.year.unique())[0],
                                                                           (np.sort(combined_data.year.unique())[
                                                                               -1]) + 1, 1)}),
                            ]),
                        ], width=9, className="p-3 mb-2 text-dark"),

                        # Tab3 First Row Second Column Here
                        dbc.Col([
                            html.Div(id='accident_count_single_year_1', )], width=3, className="p-3 mb-2 text-danger")
                    ]),

                    dbc.Row([
                        # Tab3 Second Row First Column Here
                        dbc.Col([
                            html.Div(
                                children=html.Strong("Clustering of Accidents data and display of Hot-spots on Map"),
                                style={'textAlign': 'center'}),
                            html.Br(),

                            dbc.Label("Number of accidents per cluster "),
                            dcc.Input(id="accident_number", type="number", placeholder=10),
                            html.Br(),

                            dbc.Label("Input the distance between two accidents for clustering (In mts) : "),
                            dcc.Input(id="accident_range", type="number", placeholder=100),
                            html.Br(),

                            dbc.Label("Display Accidents Spots"),
                            dcc.RadioItems(id='accidents_spots',
                                           options=[
                                               {'label': 'All', 'value': 'Y'},
                                               {'label': 'High Density', 'value': 'N'},
                                           ],
                                           value='N'
                                           ),
                            html.Br(),

                            html.Div(dbc.Button("Submit", color="light", className="mr-1"), id="clustering_submit"),
                            html.Br(),

                            html.Div(
                                [
                                    dbc.Alert(html.Div(id='clustering_status', className='col-md-3'), color="info"),
                                ]),

                            dbc.Button(
                                ["Total number of Accident Hot spots",
                                 dbc.Badge(html.Div(id="cluster_numbers"), color="light", className="ml-1")],
                                color="danger",
                            ),
                            html.Br(), html.Br(),

                            dcc.Graph(id="mapbox_figure")
                        ], width=12),

                        # Tab3 Second Row Second Column Here
                        dbc.Col([

                        ], width=0)]),

                    dbc.Row([
                        # Tab3 Third Row First Column Here
                        dbc.Col([
                            html.Div(dcc.Graph(id="selected_data_scatter"))
                        ], width=4),

                        dbc.Col([
                            html.Div(dcc.Graph(id="selected_data_histogram"))
                        ], width=4),

                        dbc.Col([
                            html.Div(dcc.Graph(id="selected_data_sunburst"))
                        ], width=4),

                        # Tab3 Third Row Second Column Here
                        dbc.Col([
                            html.Div(id='clustered_df', style={'display': 'none'})
                        ], width=0)]),

                    html.Strong('Clustering Done using DBScan Algorithm....')

                ]),

                dcc.Tab(label='Heat Map', children=[
                    # Tab4 Code here
                    html.Br(),
                    html.Div(children=html.Strong(
                        dbc.Label("Heatmap of Accidents data and display of Hot-spots on Map"),
                        style={'textAlign': 'center'})),
                    dbc.Row([
                        # Tab4 First Row First Column Here
                        dbc.Col([
                            dbc.FormGroup([
                                dbc.Label('Select Year', id='slider_value_2'),

                                dcc.Slider(id="slider_single_year_2", min=np.sort(accident_data.year.unique())[0],
                                           max=np.sort(accident_data.year.unique())[-1], step=1000,
                                           value=np.sort(accident_data.year.unique())[5],
                                           marks={x: str(x) for x in range(np.sort(accident_data.year.unique())[0],
                                                                           (np.sort(accident_data.year.unique())[
                                                                               -1]) + 1, 1)}),
                            ]),
                        ], width=9, className="p-3 mb-2 text-dark"),

                        # Tab4 First Row Second Column Here
                        dbc.Col([
                            html.Div(id='accident_count_single_year_2')], width=3, className="p-3 mb-2 text-danger")
                    ]),

                    dbc.Row([
                        # Tab4 Second Row First Column Here
                        dbc.Col([

                            html.Div(children=html.Strong("Choose specific column for filtering data for Heatmap Analysis"),
                                     style={'textAlign': 'center'}),
                            dcc.Dropdown(
                                id='columns_dropdown_heatmap', clearable=False,
                                options=[
                                    {'label': 'Police_Force', 'value': 'Police_Force'},
                                    {'label': 'Accident_Severity', 'value': 'Accident_Severity'},
                                    {'label': 'Number_of_Vehicles', 'value': 'Number_of_Vehicles'},
                                    {'label': 'Day_of_Week', 'value': 'Day_of_Week'},
                                    {'label': 'Road_Type', 'value': 'Road_Type'},
                                    {'label': 'Speed_limit', 'value': 'Speed_limit'},
                                    {'label': 'Light_Conditions', 'value': 'Light_Conditions'},
                                    {'label': 'Weather_Conditions', 'value': 'Weather_Conditions'},
                                    {'label': 'Road_Surface_Conditions', 'value': 'Road_Surface_Conditions'},
                                    {'label': 'Urban_or_Rural_Area', 'value': 'Urban_or_Rural_Area'},
                                    {'label': 'month', 'value': 'month'},
                                    {'label': 'hour', 'value': 'hour'},
                                    {'label': 'day', 'value': 'day'},
                                    {'label': 'week_in_month', 'value': 'week_in_month'},
                                    {'label': 'Vehicle_Type', 'value': 'Vehicle_Type'},
                                    {'label': 'Sex_of_Driver', 'value': 'Sex_of_Driver'},
                                    {'label': 'Age_Band_of_Driver', 'value': 'Age_Band_of_Driver'},
                                    {'label': 'Engine_Capacity_(CC)', 'value': 'Engine_Capacity_(CC)'},
                                    {'label': 'Age_of_Vehicle', 'value': 'Age_of_Vehicle'}
                                ],
                                value='Accident_Severity',
                            ),

                            dcc.Dropdown(id='columns_dropdown_heatmap_value', clearable=False, ),

                            dbc.Button('Check Hot-spots for Accidents', id='example-buttonHotspots', color='primary',
                                       style={'margin-bottom': '1em'}, block=True),

                            dcc.Graph(id="heat_map")
                        ], width=12),

                        # Tab4 Second Row Second Column Here
                        dbc.Col([

                        ], width=0)])

                ])
            ])
        ])
    ),
    html.Div('  Developed by Team C_Data.....')
])


# Callbacks here
@app.callback(
    Output('accident_count', 'children'),
    [Input('slider', 'value')])
def update_data(slider_value):
    df = accident_data
    df = df[(df['year'] >= slider_value[0]) & (df['year'] <= slider_value[1])]
    count = len(df.index)
    return 'Accident Counts in {} = {}'.format(slider_value, count)


@app.callback(
    Output('accident_count_single_year_1', 'children'),
    [Input('slider_single_year_1', 'value')])
def update_data(slider_value):
    df = combined_data
    df = df[(df['year'] == slider_value)]
    count = len(df.index)
    return 'Accident Counts in {} = {}'.format(slider_value, count)


@app.callback(
    Output('accident_count_single_year_2', 'children'),
    [Input('slider_single_year_2', 'value')])
def update_data(slider_value):
    df = accident_data
    df = df[(df['year'] == slider_value)]
    count = len(df.index)
    return 'Accident Counts in {} = {}'.format(slider_value, count)


# Call back for first Tab
@app.callback(
    Output('example_graph', 'figure'),
    [Input('dropdown', 'value')],
    [Input('slider', 'value')])
def update_figure(dropdown, slider_value):
    fig = go.Figure()
    df = accident_data
    df = df[(df['year'] >= slider_value[0]) & (df['year'] <= slider_value[1])]
    df = df.groupby(['year', 'month']).size().reset_index(name='count')
    if dropdown == 1:
        fig = px.bar(df, x=[calendar.month_abbr[(x % 12) + 1] for x in df.index], y=df['count'])
        fig.update_layout(legend_traceorder="reversed")
    else:
        fig = px.line(df, x=[calendar.month_abbr[(x % 12) + 1] for x in df.index], y=df['count'], )

    fig.update_layout(title="Accidents Data ",
                      xaxis=dict(title=go.layout.xaxis.Title(text='Months'),),
                                 # tickvals=),
                      yaxis=dict(title=go.layout.yaxis.Title(text='Aggregated Accident Counts')),
                      hovermode="x unified")

    return fig


# callback for tab1 fig1
@app.callback(
    [Output('column_values_dropdown', 'options'),
     Output('column_values_dropdown', 'value')],
    Input('column_type_dropdown', 'value'))
def set_dropdown_column_options(column_type):
    return [{'label': i, 'value': i} for i in all_options[column_type]], all_options[column_type][0]


# callback for tab1 fig2
@app.callback(
    Output('combined_agg_graph', 'figure'),
    [Input('column_values_dropdown', 'value')],
    [Input('slider', 'value')])
def update_figure(column_name, slider_value):
    df = accident_data
    df = df[(df['year'] >= slider_value[0]) & (df['year'] <= slider_value[1])]

    def time_aggregate_charts(df, column_name):

        def group_by_column_name(df, column_name, year):

            df_agg = df[df['year'] == year].groupby(column_name).Number_of_Casualties.agg(['count', 'sum'])
            if year is None:
                df_agg = df.groupby(column_name).Number_of_Casualties.agg(['count', 'sum'])
                df_agg['average'] = df_agg['sum'] / len(df.year.unique())
            df_agg.reset_index(inplace=True)
            df_agg.sort_values(by=column_name, inplace=True)

            if column_name == 'month':
                df_agg['month'] = df_agg['month'].apply(lambda x: calendar.month_abbr[x])
            if column_name == 'Day_of_Week':
                df_agg.set_index('Day_of_Week', inplace=True)
                sorter = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
                sorterIndex = dict(zip(sorter, range(len(sorter))))
                df_agg['Day_id'] = df_agg.index
                df_agg['Day_id'] = df_agg['Day_id'].map(sorterIndex)
                df_agg.sort_values('Day_id', inplace=True)
                df_agg.reset_index(inplace=True)

            return df_agg

        year_list = df['year'].unique().tolist() + [None]
        list_of_traces = []
        colors = ['darkmagenta', 'deeppink', 'lavender', 'lightsteelblue', 'orchid', 'navy', 'forestgreen',
                  'greenyellow', 'silver', 'darkslategrey', 'red', 'yellow', 'magenta', 'orange']

        for year in range(len(year_list)):
            if year_list[year] is None:
                name_agg = group_by_column_name(df, column_name, year_list[year])
                data_agg = go.Scatter(x=name_agg[column_name], y=name_agg['average'], mode="lines+markers",
                                      name='Overall Average',
                                      line=dict(color='darkslategrey', width=4))
            else:
                name_agg = group_by_column_name(df, column_name, year_list[year])
                data_agg = go.Scatter(x=name_agg[column_name], y=name_agg['sum'], mode="lines+markers",
                                      name='Agg Count for ' + str(year_list[year]),
                                      line=dict(color=colors[year], width=2, dash='dashdot'),
                                      hoverlabel=dict(namelength=-1))
            list_of_traces.append(data_agg)

        # Data visualization
        if column_name == 'month':
            content = 'by Months of the year'
            tk = ''
        elif column_name == 'hour':
            content = 'during Day and Night'
            tk = 1
        elif column_name == 'Day_of_Week':
            content = 'by weekdays'
            tk = ''
        elif column_name == 'week_in_month':
            content = 'by weeks in a month'
            tk = ''
        else:
            content = 'by ' + str(column_name)
            tk = ''

        layout = go.Layout(title='Road Accidents ' + content +
                                 ' in UK between ' + str(year_list[0]) + ' - ' + str(year_list[-2]),
                           xaxis=dict(title='<b> ' + column_name + ' <b>', titlefont=dict(size=16, color='#7f7f7f'),
                                      tickfont=dict(size=15, color='darkslateblue')),
                           yaxis=dict(title='<b> Number of Casualties <b>', titlefont=dict(size=16, color='#7f7f7f'),
                                      tickfont=dict(size=15, color='darkslateblue')))

        fig = go.Figure(data=list_of_traces, layout=layout)
        fig.update_xaxes(dtick=tk)
        fig.update_layout(hovermode="x unified")
        return fig

    graph = time_aggregate_charts(df, column_name)
    return graph


# callback for tab1 fig3
@app.callback(
    Output('treemap', 'figure'),
    [Input('slider', 'value')],
    [Input('columns_dropdown_treemap', 'value')])
def update_treemap(slider_value, columns):
    df = combined_data
    df = df[(df['year'] >= slider_value[0]) & (df['year'] <= slider_value[1])]
    df = df.sample(frac=0.1, replace=False, random_state=1)
    # df = df.dropna(axis=0)
    for col in columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype(int)

    df['all'] = ''
    path = ['all'] + columns
    fig = px.treemap(df, path=path, values='Number_of_Casualties')

    return fig


# callback for tab1 fig4
@app.callback(
    Output('correlation', 'figure'),
    [Input('slider', 'value')])
def update_figureC(slider_value):
    cd = combined_data

    cd['Date'] = pd.to_datetime(cd['Date'], dayfirst=True)

    cd = cd[(cd['year'] >= slider_value[0]) & (cd['year'] <= slider_value[1])]

    corr = cd.corr()

    data = [
        go.Heatmap(
            z=corr,
            x=corr.columns,
            y=corr.index,
            colorscale=color_scale
        )
    ]

    layout = go.Layout(
        height=600,
        width=600,
        # title='Correlation between Features',
        xaxis=dict(ticks='', nticks=36, automargin=True),
        yaxis=dict(ticks='', automargin=True)
    )

    fig = go.Figure(data=data, layout=layout)
    # fig.update_layout(
    #     plot_bgcolor=colors['background'],
    #     paper_bgcolor=colors['background'],
    #     font_color=colors['text']
    # )
    return fig


# Callback for second tab-Time series prediction
@app.callback(
    [Output('RMSE_Value', 'children'),
     Output('time_series_figure', 'figure')],
    [Input('time_aggregate', 'value')],
    [Input('grouping_col', 'value')])
def update_time_series_figure(time_range, column):
    model_error_total = 0
    fig = go.Figure()
    df = accident_data
    df = df.groupby(['Date', column]).size().reset_index(name='count')
    df['Date'] = pd.to_datetime(df['Date'])
    colors = ['#389393', '#ea2c62', '#fecd1a', '#99a8b2']

    df1 = df.copy(deep=True)
    df1.drop(column, axis=1, inplace=True)
    df1.set_index('Date', inplace=True)
    df1 = df1.resample(time_range).sum()
    fig.add_trace(go.Scatter(x=df1.index, y=df1.iloc[:, -1], mode='lines', marker_color=colors[0],
                             name='Total_Accidents', hoverlabel=dict(namelength=-1)))

    if time_range != 'Y':
        df_new, df_train_size, model_error_total = predict_next(df1, 24, time_range)
        df_new.drop(index=df1.index[:df_train_size], inplace=True)
        fig.add_trace(go.Scatter(x=df_new.index, y=df_new.iloc[:, -1], mode='lines', name="Prediction->Total_Accidents",
                                 marker_color='red', hoverlabel=dict(namelength=-1)))

    for val in np.sort(df[column].unique()):
        df1 = df[df[column] == val]
        df1.drop(column, axis=1, inplace=True)
        df1.set_index('Date', inplace=True)
        df1 = df1.resample(time_range).sum()
        fig.add_trace(go.Scatter(x=df1.index, y=df1.iloc[:, -1], mode='lines', name=str(column) + '=' + str(val),
                                 hoverlabel=dict(namelength=-1)))
        if time_range != 'Y':
            df_new, df_train_size, model_error = predict_next(df1, 24, time_range)
            df_new.drop(index=df1.index[:df_train_size], inplace=True)
            fig.add_trace(go.Scatter(x=df_new.index, y=df_new.iloc[:, -1], mode='lines',
                                     name='Prediction->' + str(column) + '=' + str(val), marker_color='red',
                                     showlegend=False, hoverlabel=dict(namelength=-1)))

    fig.update_layout(title="Accidents Data ",
                      barmode='overlay',
                      xaxis=dict(title=go.layout.xaxis.Title(text='Date')),
                      yaxis=dict(title=go.layout.yaxis.Title(text='Aggregated Count Values')),
                      # height=400,
                      hovermode="x unified")

    return "Root Mean Square Error for ARIMA model's Training is :{}".format(model_error_total), fig


# Callback for third tab-Clustering of accidents data
@app.callback([
    Output('cluster_numbers', 'children'),
    Output('mapbox_figure', 'figure'),
    Output('clustering_status', 'children'),
    Output('clustered_df', 'children')],
    [Input('clustering_submit', 'n_clicks')],
    [State('accident_range', 'value')],
    [State('accidents_spots', 'value')],
    [State('slider_single_year_1', 'value')],
    [State('accident_number', 'value')],
)
def find_clusters_dbscan(n_clicks, distance, accidents_spots, slider_value, accident_number):
    # df = accident_data
    df = combined_data  # here just use combined_data instead of accident_data
    df = df[(df['year'] == slider_value)]
    cluster_numbers = 0

    if accidents_spots == 'Y':
        fig = plot_map(df)
        return ['{}'.format(0), fig, 'No clustering done', '']

    elif accidents_spots == 'N':
        if distance is None:
            distance = 100
        if accident_number is None:
            accident_number = 10

        # calling dbscan method for clustering and finding labels
        df_clustered = clustering(df, distance, accident_number)
        cluster_numbers = df_clustered['Cluster'].unique().size - 1

        if cluster_numbers > 0:
            df_clustered = df_clustered[df_clustered['Cluster'] != -1]
            fig = plot_map(df_clustered, color='Y')
            return [f'{cluster_numbers}', fig, 'Clustering Completed', df_clustered.to_json(date_format='iso', orient='split')]

        fig = plot_map(df)
        return ['{}'.format(cluster_numbers), fig, 'Clustering Completed: No Clusters found for given parameters',
                df_clustered.to_json(date_format='iso', orient='split')]


def plot_sunburst(newCluster):
    figure = px.sunburst(newCluster, path=['Sex_of_Driver', 'Speed_limit', 'Accident_Severity', 'Day_of_Week'],
                         values='Number_of_Vehicles', color='Weather_Conditions',
                         color_discrete_map={'(?)': 'green', 'Lunch': 'gold', 'Dinner': 'darkblue'})
    return figure


def no_data_selected():
    return {"layout": {
        "xaxis": {
            "visible": False
        },
        "yaxis": {
            "visible": False
        },
        "annotations": [
            {
                "text": "No Data Selected",
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 28
                }
            }
        ]
    }}


@app.callback(
    Output('selected_data_sunburst', 'figure'),
    [Input('mapbox_figure', 'selectedData')],
    [State('clustered_df', 'children')]
)
def populate_seleted_graph(selectedData, clustered_df):
    if clustered_df is None:
        return no_data_selected()

    df_clustered = pd.read_json(clustered_df, orient='split')
    # print("Selected Data: ", selectedData)
    # print(df_clustered)
    selected_Clustered = selectedData['points'][0]['marker.color']

    newCluster = df_clustered.loc[df_clustered['Cluster'] == selected_Clustered]
    # print(newCluster)
    # print(df_clustered.columns)
    fig = plot_sunburst(newCluster)
    return fig


@app.callback(
    Output('selected_data_scatter', 'figure'),
    [Input('mapbox_figure', 'selectedData')],
    [State('clustered_df', 'children')]
)
def populate_seleted_graph(selectedData, clustered_df):
    if clustered_df is None:
        return no_data_selected()

    df_clustered = pd.read_json(clustered_df, orient='split')
    selected_Clustered = selectedData['points'][0]['marker.color']
    newCluster = df_clustered.loc[df_clustered['Cluster'] == selected_Clustered]

    fig = px.scatter(newCluster, x="Age_of_Vehicle", y="Age_Band_of_Driver", color="Sex_of_Driver",
                     size='Engine_Capacity_(CC)', hover_data=['Road_Type', 'Light_Conditions'],
                     labels=dict(Age_of_Vehicle="Age of the Vehicle in years",
                                 Age_Band_of_Driver="Age Band of the Driver"))

    return fig


@app.callback(
    Output('selected_data_histogram', 'figure'),
    [Input('mapbox_figure', 'selectedData')],
    [State('clustered_df', 'children')]
)
def populate_seleted_graph(selectedData, clustered_df):
    if clustered_df is None:
        return no_data_selected()

    df_clustered = pd.read_json(clustered_df, orient='split')
    selected_Clustered = selectedData['points'][0]['marker.color']
    newCluster = df_clustered.loc[df_clustered['Cluster'] == selected_Clustered]

    fig = px.histogram(newCluster, x="Day_of_Week",
                       title='Histogram of Accidents over week',
                       labels={'Day_of_Week': 'Day of the week', 'count': 'Number of Accidents'},
                       # can specify one label per df column
                       opacity=0.8,
                       log_y=True,  # represent bars with log scale
                       color_discrete_sequence=['indianred']  # color of histogram bars
                       )

    return fig


# Callback for fourth tab-Heatmap
@app.callback(
    [Output('columns_dropdown_heatmap_value', 'options'),
     Output('columns_dropdown_heatmap_value', 'value')],
    Input('columns_dropdown_heatmap', 'value'))
def set_dropdown_column_options(column):
    df = combined_data
    df_new = df[column]
    unique_vals = df_new.unique()
    return [{'label': i, 'value': i} for i in unique_vals], unique_vals[0]

@app.callback(Output('heat_map', 'figure'),
              [Input('example-buttonHotspots', 'n_clicks')],
              [State('columns_dropdown_heatmap', 'value')],
              [State('columns_dropdown_heatmap_value', 'value')],
              [State('slider_single_year_2', 'value')], )
def update_figureHeatMap(n_clicks, column, column_value, slider_value):
    head_df = combined_data
    head_df = head_df[(head_df['year'] == slider_value)]
    head_df1 = head_df.loc[head_df[column] == column_value]
    fig = plot_heatmap(head_df1)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
