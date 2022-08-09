import pandas as pd
import numpy as np

# plot
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots


# dashboards
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc # for dashboard themes

# time
from datetime import date

data = pd.read_csv('timesData.csv')

def capitalize(string):
    return string.capitalize()
data.columns = map(capitalize, data.columns)

data['World_rank'] = data['World_rank'].str.replace('=', '')
dataAll = data
data = data[~data['World_rank'].str.contains('-')]

for i in ['World_rank','International','Total_score','Income']:
    data[i] = pd.to_numeric(data[i],errors = 'coerce')

df = pd.DataFrame({'label':data['Country'],'value':data['Country']})
df = df.drop_duplicates()
d_records = df.to_dict('records')


# def create_records(year):
#     data = data[data['Year'] == year]
#     df = pd.DataFrame({'label':data['Country'],'value':data['Country']})
#     df = df.drop_duplicates()
#     d_records = df.to_dict('records')
#     return d_records

metrics = ['Teaching', 'International','Research', 'Citations', 'Income']
df2 = pd.DataFrame({'label':metrics,'value':metrics})
m_records = df2.to_dict('records')
top10country = list(data['Country'].value_counts().head(10).index)

def blank_fig():
    fig = go.Figure(go.Scatter(x=[], y = []))
    fig.update_layout(template = None)
    fig.update_xaxes(showgrid = False, showticklabels = False, zeroline=False)
    fig.update_yaxes(showgrid = False, showticklabels = False, zeroline=False)
    return fig


def country_uni_num(year):
    country_uni_num = pd.DataFrame({'Number of universities in top 200': \
                                    data[data['Year'] == year]['Country'].value_counts()})
    return country_uni_num

def country_uni_rank(year):
    country_uni_rank = data[(data['Country'].isin(top10country)) & \
                            (data['Year'] == year)][['World_rank', 'Country']]
    return country_uni_rank

def radar(i):
    r=data.loc[(data['Year']==i),['University_name','Teaching','International','Research','Citations','Income','Total_score']].head(10).reset_index()
    r.drop('index',axis=1, inplace=True)
    return r

def state_uni_rank(year):
    df=data[data['Year'] == year][['World_rank', 'Country']]
    df.set_index('Country',inplace=True)
    for i in df.index:
        if i in ['United States of America','Canada']:
            df.loc[i,'State']='North America'
        if i in ['Germany','United Kingdom','France','Netherlands','Sweden','Switzerland','Finland','Denmark','Belgium','Austria','Spain','Republic of Ireland','Norway']:
            df.loc[i,'State']='Europe'
        if i in ['Australia','New Zealand']:
            df.loc[i,'State']='Oceania'
        if i in ['Japan','China','Hong Kong','South Korea', 'Taiwan','Turkey','Singapore']:
            df.loc[i,'State']='Asia'
        if i in ['Egypt','South Africa']:
            df.loc[i,'State']='Africa'
    return df

def country_spend_line():
    data = pd.read_csv('public spending.csv')
    colorList=['rgb(24, 78, 119)',
            'rgb(30, 96, 145)',
            'rgb(26, 117, 159)',
            'rgb(22, 138, 173)',
            'rgb(52, 160, 164)',
            'rgb(82, 182, 154)',
            'rgb(118, 200, 147)',
            'rgb(153, 217, 140)',
            'rgb(181, 228, 140)',
            'rgb(217, 237, 146)']
    fig = px.line(data, x='Year', 
                y='Public spending on tertiary education(% of GDP)', 
                color='Country',markers=True,title = 'Public Spending on Tertiary Education (% of GDP)',color_discrete_sequence = colorList) # px.colors.sequential.Purples)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=False
        ),
        autosize=True,
        plot_bgcolor='white'
    )
    return fig

data1 = data.dropna(subset=['International_students'])
data1['International'] = data1['International_students'].str.replace('%','').astype('int')/100
data1['Local'] = 1 - data1['International']

data2 = data.dropna(subset=['Female_male_ratio'])
data2['Female_male_ratio'] = data2['Female_male_ratio'].astype('str')
data2['Female'] = data2['Female_male_ratio'].map(lambda x:x.split(' :')[0])
data2['Male'] = data2['Female_male_ratio'].map(lambda x: x.split(' : ')[1])

data3 = data
data3['Student'] = data3['Student_staff_ratio']*10
data3['Teacher'] = 10
data3 = data3[~data3['Student_staff_ratio'].isna()]
data3['Student'] = data3['Student'].astype('int')

df = pd.DataFrame({'label':data1['University_name'],'value':data1['University_name']})
df = df.drop_duplicates()
u_records = df.to_dict('records')

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])

app.layout = html.Div([
    html.Div([
        html.H1('World University Ranking'),
        html.P("Helping Students and Schools to Embrace a Better Future",style={'color':'rgb(222,226,230)'})
        ], 
        style = {'padding' : '50px' , 
                'backgroundColor' : '#3aaab2'}
    ),
    html.Br(),

    dcc.Tabs([
        dcc.Tab([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                dcc.Graph(id='map')],style = {'padding': '0px 200px 0px 200px'}
            ),
            html.Br(),
            html.Div(
                dcc.Slider(
                    id='map-year-slider',
                    min=2011,
                    max=2016,
                    value=2012,
                    marks={str(year): str(year) for year in range(2011,2017)}
                )
            ,style = {'padding': '0px 250px 0px 250px'}),
            html.Div([
                dcc.Graph(id='state-boxplot')
                ]
            ,style = {'padding': '0px 200px 0px 200px'}),
            ],label='Overview'),

        dcc.Tab([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Br(),
            html.P([
                html.Label("Choose a metric:"),
                dcc.Dropdown(
                    id = 'metrics_dropdown',
                    options= m_records,
                    multi=False,
                    value = 'Teaching')
                ],
                style = {
                            'fontSize' : '20px',
                            'padding': '0px 250px 0px 250px',
                            'display': 'inline-block'}),
            html.Div([
                dcc.Graph(id='metrics-boxplot')
            ],style = {'padding': '0px 200px 0px 200px'}),
            html.Div([
                dcc.Slider(
                    id='year-slider2',
                    min=2011,
                    max=2016,
                    value=2012,
                    marks={str(year): str(year) for year in range(2011,2017)},
                )
            ],style = {'padding': '0px 300px 0px 300px'}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                dcc.Graph(id='ranking-boxplot')
            ],style = {'padding': '0px 200px 0px 200px'}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div([
                dcc.Graph(figure = country_spend_line())
            ],style = {'padding': '0px 200px 0px 200px'}),
        ], label='Countries'),

        dcc.Tab([
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div(
                dcc.Slider(
                    id='year-slider3',
                    min=2011,
                    max=2016,
                    value=2012,
                    marks={str(year): str(year) for year in range(2011,2017,1)},
                ),style = {'padding': '0px 300px 0px 300px'}
            ),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Div(
                dcc.Graph(id='radar')
                ),
            html.Br(),
            html.Br(),
            html.P([
                html.Label("Choose a country:"),
                dcc.Dropdown(
                    id = 'country_dropdown',
                    options= d_records,
                    multi=False,
                    value = 'United States of America')
                ],
                style = {'width': '400px',
                            'fontSize' : '20px',
                            'padding-left' : '100px',
                            'display': 'inline-block'}),
            dcc.Graph(id='graph-with-slider'),  
            html.P([
                html.Label("Choose a University:"),
                dcc.Dropdown(
                    id = 'university_dropdown',
                    multi=False,
                    value = 'California Institute of Technology')
                ],
                style = {'width': '600px',
                            'fontSize' : '20px',
                            'padding-left' : '100px',
                            'display': 'inline-block'}), 
            # dcc.Graph(figure = piechart_staff('United States of America', 2012, 'California Institute of Technology')),          
            html.Div([
                html.Div(
                    dcc.Graph(id='Pie-International-Student'),
                 style={'width': '33.3%', 'display': 'inline-block'}),
                html.Div(
                    dcc.Graph(id='Pie-Teacher-Ratio'),
                    style={'width': '33.3%', 'display': 'inline-block'}
                ),           
                html.Div(
                    dcc.Graph(id='Pie-Male-Ratio'),
                    style={'width': '33.3%', 'display': 'inline-block'}
                )                
            ],style = {'padding': '0px 100px 0px 100px'}),
            html.Div(
                dcc.Graph(id='university_linechart',figure = blank_fig()),style = {'padding': '0px 200px 0px 200px'}
            )
            
              
        ], label='Universities')
    ]),
])

@app.callback(
    [Output('map', 'figure'),
    Output('state-boxplot', 'figure')],
    [Input('map-year-slider', 'value')]
)
def update_output(value):
    df = country_uni_num(value)
    fig1 = go.Figure(data=go.Choropleth(
        locations = df.index,
        locationmode = 'country names',
        z = df['Number of universities in top 200'],
        colorscale = 'Teal',
        marker_line_color='white',
        marker_line_width=0.7,
        showscale = False
        ))
    fig1.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        margin=dict(t=0, b=0, l=0, r=0))
    df2 =state_uni_rank(value)
    fig2 = go.Figure()
    fig2.add_trace(go.Box(y=df2['World_rank'], x=df2['State'],
                        marker_color = '#3aaab2'))
    fig2.update_layout(
        title=f"Top 200 Universities' Distribution in Continents ({value})",
        xaxis_tickfont_size=14,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)'),
            categoryorder = 'array'),    
        plot_bgcolor='white',
        showlegend = False)
    return fig1,fig2

@app.callback(
        Output('radar', 'figure'),
        Input('year-slider3', 'value'))
def update_figure(value):
    df=radar(value)
    color='#3aaab2'
    categories=list(df)[1:]
    fig = make_subplots(
        rows=3, cols=4, specs=[[{'type': 'polar'}]*4]*3,horizontal_spacing=0.16,vertical_spacing=0.05,subplot_titles=(df['University_name'].values.flatten().tolist()))
    fig.add_trace(go.Scatterpolar(
    r=df.loc[0].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar1",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[0,'University_name'],
    ),1,1)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[1].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar2",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[1,'University_name']
    ),1,2)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[2].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar3",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[2,'University_name']
    ),1,3)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[3].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar4",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[3,'University_name']
    ),1,4)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[4].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar5",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[4,'University_name']
    ),2,1)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[5].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar6",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[5,'University_name']
    ),2,2)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[6].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar7",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[6,'University_name']
    ),2,3)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[7].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar8",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[7,'University_name']
    ),2,4)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[8].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar9",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[8,'University_name']
    ),3,1)
    fig.add_trace(go.Scatterpolar(
    r=df.loc[9].drop('University_name').values.flatten().tolist(),
    theta = categories,
    subplot = "polar10",
    fill = 'toself',
    marker=dict(color=color),
    name= df.loc[9,'University_name']
    ),3,2)
    fig.update_layout(
    polar1 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100],
        )),
    polar2 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar3 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )), 
    polar4 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),  
    polar5 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar6 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar7 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar8 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar9 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    polar10 = dict(
        radialaxis = dict(
            visible = False,
            color = 'grey',
            range = [0, 100]
        )),
    
    showlegend = False
    )
    fig.update_layout(height=1000, width=1400)
    fig.update_annotations(font_size=15,yshift=5)
    fig.update_layout(
        title={
            'text': f'Times Higher Education World University Ranking Top 10 ({value})',
            'y':0.99,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},title_font_size=25)
    return fig

@app.callback(
    Output('graph-with-slider', 'figure'),
    [Input('year-slider3', 'value'),
    Input('country_dropdown', 'value')])
def update_table(year,country):
    df = data[(data['Year'] == year) & (data['Country'] == country)]
    fig = go.Figure(data=go.Table(columnwidth = [15,50,13,17,13,13,12,15],
        header=dict(values=list(df[['World_rank','University_name',
                                    'Teaching', 'International','Research', 
                                    'Citations', 'Income','Total_score']]),
                    fill_color='rgb(98, 182, 203)',
                    font=dict(color='white', size=22),
                    height=36,
                    align='center',
                   ),
        cells=dict(values=[df.World_rank, df.University_name, 
                           df.Teaching, df.International, df.Research,
                          df.Citations,df.Income,df.Total_score],
                   fill_color='rgb(190, 233, 232)',
                   height=32,
                   align='center',
                  font=dict(size=18)))
        
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        title={
            'text': f'Top 200 Universities in {country} ({year})',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_size=22
    )
    return fig

@app.callback(
    Output('metrics-boxplot', 'figure'),
    [Input('year-slider2', 'value'),
    Input('metrics_dropdown', 'value')])
def update_metrics_boxplot(year,metrics):
    df = data[(data['Country'].isin(top10country)) & (data['Year'] == year)]
    fig = px.box(df, x=metrics, y="Country")

    fig.update_layout(
    title=f'{metrics} Score Distribution for Top 10 Countries ({year})',
    xaxis_tickfont_size=14,
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)'
        )
       ), 
    yaxis={'categoryorder':'array', 'categoryarray':top10country},  
    plot_bgcolor='white',
    showlegend = False)
    fig.update_traces(orientation='h',marker_color = '#3aaab2')
    return fig

@app.callback(
    Output('ranking-boxplot', 'figure'),
    Input('year-slider2', 'value'))
def update_ranking_boxplot(year):
    df = country_uni_rank(year)
    fig = go.Figure()
    fig.add_trace(go.Box(y=df['World_rank'], x=df['Country'],
                        marker_color = '#3aaab2'))
    fig.update_layout(
        title=f'Top 200 Universities Distribution in Top 10 Countries ({year})',
        xaxis_tickfont_size=14,
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)'),
            categoryorder = 'array',
            categoryarray = top10country),    
        plot_bgcolor='white',
        showlegend = False)
    return fig

@app.callback(
    Output('Pie-International-Student', 'figure'),
    [Input('year-slider3', 'value'),
    Input('country_dropdown', 'value'),
    Input('university_dropdown', 'value')])
def piechart(year,country,university):
    df = data1[(data1['Country'] == country) 
                & (data1['Year'] == year)
                & (data1['University_name'] == university)]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False),
            yaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False)
        )
    else:
        df = df[['International','Local']].T
        df.columns = [university]
        fig = px.pie(df,values=university, names=df.index
                    ,color_discrete_sequence=px.colors.sequential.Teal
                    ,hover_data=[university], labels={university:'ratio'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend = False
        )
    return fig

@app.callback(
    Output('Pie-Male-Ratio', 'figure'),
    [Input('year-slider3', 'value'),
    Input('country_dropdown', 'value'),
    Input('university_dropdown', 'value')])
def piechart_female(year,country,university):
    df = data2[(data2['Country'] == country) 
              & (data2['Year'] == year)
              & (data2['University_name'] == university)]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False),
            yaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False)
        )
    else:
        df = df[['Female','Male']].T
        df.columns = [university]
        fig = px.pie(df,values=university, names=df.index
                    ,color_discrete_sequence=px.colors.sequential.Teal
                    ,hover_data=[university], labels={university:'ratio'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend = False
        )
    return fig

@app.callback(
    Output('Pie-Teacher-Ratio', 'figure'),
    [Input('year-slider3', 'value'),
    Input('country_dropdown', 'value'),
    Input('university_dropdown', 'value')])
def piechart_staff(year,country,university):
    df = data3[(data3['Country'] == country) 
              & (data3['Year'] == year)
              & (data3['University_name'] == university)] 
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False),
            yaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False)
        )
    else:
        df = df[['Student','Teacher']].T
        df.columns = [university]
        fig = px.pie(df,values=university, names=df.index
                    ,color_discrete_sequence=px.colors.sequential.Teal
                    ,hover_data=[university], labels={university:'ratio'})
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            showlegend = False
        )
    return fig



@app.callback(
    Output('university_dropdown', 'options'),
    [Input('year-slider3', 'value'),
    Input('country_dropdown', 'value')])
def select_country(year,country):
    df = data[(data['Country'] == country) & (data['Year'] == year)]
    df = pd.DataFrame({'label':df['University_name'],'value':df['University_name']})
    df = df.drop_duplicates()
    u_records = df.to_dict('records')
    return u_records



@app.callback(
    Output('university_linechart', 'figure'),
    [Input('country_dropdown', 'value'),
    Input('university_dropdown', 'value')])
def linechart(country,university):       
    df = data[(data['Country'] == country)&(data['University_name'] == university)]
    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='white',
            xaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False),
            yaxis=dict(
                showline=False,
                showgrid=False,
                showticklabels=False)
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["Year"], y=df["World_rank"],
                        mode='lines+markers',line = dict(color='#3aaab2', width=4),
                        marker=dict(size=10)))
        fig.update_layout(
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
                linecolor='rgb(204, 204, 204)',
                linewidth=2,
                ticks='outside',
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=True,
            ),
            plot_bgcolor='white',
            xaxis_title='Year',
            yaxis_title='Ranking',
            title=f"{university}'s Ranking Overtime",
        )
    
    return fig


if __name__ == '__main__':
    app.run_server(debug = True)


    