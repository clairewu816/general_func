import plotly.plotly as py
from plotly.graph_objs import *


def map_two_sources(df, title):
    """
    Plot two sources of geo-data into one us-map
    :param df: concat two source with cols=['latitude', 'longitude', 'text', 'tag']
    :param title: graph title
    :return: Figure object and ready to use py.plot() to plot
    """
    scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"], \
           [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]

    data = [dict(
        type='scattergeo',
        locationmode='USA-states',
        lon=df['longitude'],
        lat=df['latitude'],
        text=df['text'],
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale=scl,
            cmin=0,
            color=df['tag'],
            cmax=df['tag'].max()
        ))]

    layout = dict(
        title=title,
        colorbar=True,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
    )

    return dict(data=data, layout=layout)


def map_states_dots(states_df, dots_df, title):
    """
    Plot states area and dots into the same US-map
    :param states_df: cols = ['state_abbrev', 'tag'] -- 'tag' as values
    :param dots_df: cols = ['latitude', 'longitude','total_score' , 'text']
    :param title: graph title
    :return: Figure object and ready to use py.plot() to plot
    """
    scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"], \
           [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]

    trace1 = dict(
        type='scattergeo',
        locationmode='USA-states',
        lon=dots_df['longitude'],
        lat=dots_df['latitude'],
        text=dots_df['text'],
        mode='markers',
        marker=dict(
            size=12,
            opacity=0.8,
            reversescale=True,
            autocolorscale=False,
            line=dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale=scl,
            cmin=0,
            color=dots_df['total_score'],
            cmax=dots_df['total_score'].max(),
            colorbar=dict(
                title="total leo_score"
            )
        ))

    trace2 = dict(
        type='choropleth',
        locations=states_df['state_abbrev'],
        z=states_df['tag'],
        locationmode='USA-states',
        text=states_df['state_abbrev'],
        marker=dict(
            line=dict(
                color='rgb(255,255,255)',
                width=2
            )),
    )

    data = Data([trace2, trace1])
    layout = dict(
        title=title,
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showland=True,
            landcolor="rgb(250, 250, 250)",
            subunitcolor="rgb(217, 217, 217)",
            countrycolor="rgb(217, 217, 217)",
            countrywidth=0.5,
            subunitwidth=0.5
        ),
    )
    return Figure(data=data, layout=layout)
