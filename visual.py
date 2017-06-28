import plotly.plotly as py
from plotly.graph_objs import *
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_timeline(ax, y0, y1, md, c='grey'):
    """
    Plot vertical time lines
    :param y0: start year
    :param y1: end year, not included
    :param md: e.g. '06-01'
    :param c: color
    """
    xpos = [pd.to_datetime(str(x) + '-' + md) for x in range(y0, y1)]
    for i in xpos:
        ax.axvline(i, color=c)


def plot_subplots(y1, y2, y3, titles):
    """
    Plot subplots with prices and temperature
    if only one price, leave y2 as ''
    titles=[y1_label, y2_label, plot_title]
    :return:
    """
    ylabels = ['price', 'temperature']
    f, axes = plt.subplots(2, 1, figsize=(17, 8))

    plot_timeline(axes[0], 2011, 2017, '03-01', 'black')
    plot_timeline(axes[0], 2011, 2017, '06-01')
    axes[0].plot(y1, 'b', label=titles[0])
    if isinstance(y2, pd.Series):
        axes[0].plot(y2, 'g', label=titles[1])
    axes[0].set_ylabel(ylabels[0])
    axes[0].set_title(titles[2])
    axes[0].legend(loc='upper left')
    axes[0].grid(True)

    plot_timeline(axes[1], 2011, 2017, '03-01', 'black')
    plot_timeline(axes[1], 2011, 2017, '06-01')
    axes[1].plot(y3, 'r')
    axes[1].set_ylabel(ylabels[1])
    plt.grid(True)


def plot_subplots2(y1, y2, y3, ylabels, title):
    """
    Comprehensive version of plot_subplots() by adding a second axis in the first subplot
    """
    f, axes = plt.subplots(2, 1, figsize=(17, 8))

    plot_timeline(axes[0], 2011, 2017, '03-01', 'black')
    plot_timeline(axes[0], 2011, 2017, '06-01')
    axes[0].plot(y1, 'b', label=ylabels[0])
    axes[0].set_ylabel(ylabels[0], color='b')
    axes[0].tick_params('y', colors='b')

    ax2 = axes[0].twinx()
    ax2.plot(y2, 'g', label=ylabels[1])
    ax2.set_ylabel(ylabels[1], color='g')
    ax2.tick_params('y', colors='g')

    axes[0].set_title(title)
    axes[0].grid(True)

    plot_timeline(axes[1], 2011, 2017, '03-01', 'black')
    plot_timeline(axes[1], 2011, 2017, '06-01')
    axes[1].plot(y3, 'r')
    axes[1].set_ylabel(ylabels[2])
    axes[1].grid(True)


def plot_hist_line_charts_together(hist_df, hist_axises, line_df, line_axises, ts_flag=False):
    """
    Plot hist and line charts together
    :param hist_df:
    :param hist_axises: [x_axis, y_axis]
    :param line_df:
    :param line_axises: [x_axis, y_axis]
    :param ts_flag: if those are time series
    """
    hist_x_axis, hist_y_axis = hist_axises
    line_x_axis, line_y_axis = line_axises
    if ts_flag:
        hist_df[hist_x_axis] = hist_df[hist_x_axis].astype(str)
        line_df[line_x_axis] = line_df[line_x_axis].astype(str)
    ax = line_df.plot(x=line_x_axis, y=line_y_axis, linestyle='-', marker='o')
    hist_df.plot(x=hist_x_axis, y=hist_y_axis, kind='bar', ax=ax)
