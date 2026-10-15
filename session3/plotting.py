# numpy is a library used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# set the style of the plotly figures
pio.templates.default = "none"

# import some custom functions we wrote
# the colours every notebook, the slides and the task share
from session3 import palette
from session3 import fitting
from session3 import loading


def plot_schedule(ID, *df):
  '''
  Plots the experimental schedule using plotly.

    Parameters:
        ID (int): The participant ID.
        df (pandas.DataFrame): optional, a dataframe containing fitted parameters.

  '''

  trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded = loading.load_blain(ID)

  # compute number of trials
  nTrials = len(opt1Rewarded)

  # create subplots
  if not df:
    fig = make_subplots(rows=2, cols=1)
  else:
    fig = make_subplots(rows=3, cols=1)
    nTrials = len(opt1Rewarded)
    probOpt1    = np.zeros(nTrials, dtype = float)
    choiceProb1 = np.zeros(nTrials, dtype = float)
    utility1    = np.zeros(nTrials, dtype = float)
    utility2    = np.zeros(nTrials, dtype = float)

    if any(df[0].columns == 'alphaStable'):
        if any(df[0].columns == 'omega'):
            if ID < 37:
                _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alphaStable[ID],   df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
                _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alphaVolatile[ID], df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
            else:
                _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alphaVolatile[ID], df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
                _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alphaStable[ID],   df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
        else:
            if ID < 37:
                _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alphaStable[ID],   df[0].beta[ID])
                _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alphaVolatile[ID], df[0].beta[ID])
            else:
                _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alphaVolatile[ID], df[0].beta[ID])
                _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alphaStable[ID],   df[0].beta[ID])
    else:
        # with a single learning rate the block order does not matter
        if any(df[0].columns == 'omega'):
            _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alpha[ID], df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
            _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alpha[ID], df[0].beta[ID], df[0].omega[ID], utility_function = fitting.additive_utility)
        else:
            _, probOpt1[0:80],   choiceProb1[0:80],   utility1[0:80],   utility2[0:80]   = fitting.simulate_RL_model(opt1Rewarded[0:80],   magOpt1[0:80],   magOpt2[0:80],   df[0].alpha[ID], df[0].beta[ID])
            _, probOpt1[80:160], choiceProb1[80:160], utility1[80:160], utility2[80:160] = fitting.simulate_RL_model(opt1Rewarded[80:160], magOpt1[80:160], magOpt2[80:160], df[0].alpha[ID], df[0].beta[ID])


  # plot trueProbability as a line
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = trueProbability,
          mode = 'lines',
          line = dict(color=palette.TRUTH, dash='dash'),
          name = "true probability"
      ))

  # check if we should plot probOpt1, and if so plot it as a line
  if df:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = probOpt1,
            mode = 'lines',
            line = dict(color=palette.OPTION1),
            name = "RL model probability"
        ))

  # check if we should plot choiceProb1, and if so plot it as a scatterplot
  if df:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choiceProb1,
            mode = 'markers',
            marker = dict(size = 5, symbol='x', color=palette.OPTION1),
            name = "choice probability"
        ))

  # get correct and incorrect choices
  correct_choices   = np.array([choice1[i] if (opt1Rewarded[i] and     choice1[i]) or  (not opt1Rewarded[i] and not choice1[i]) else np.nan for i in range(nTrials)])
  incorrect_choices = np.array([choice1[i] if (opt1Rewarded[i] and not choice1[i]) or  (not opt1Rewarded[i] and     choice1[i]) else np.nan for i in range(nTrials)])

  # plot choices as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = correct_choices,
          mode = 'markers',
          marker = dict(size = 6, opacity=0.8, color=palette.by_option(choice1)),
          name = "correct choice"
      ))

  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = incorrect_choices,
          mode = 'markers',
          marker = dict(size = 6, opacity=0.8, color=palette.by_option(choice1), symbol='x'),
          name = "incorrect choice"
      ))

  # label the axes
  fig.update_layout(xaxis_title="", yaxis_title="option 1 rewarded?")

  # plot reward magnitudes as lines
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = magOpt1,
          mode = 'lines',
          line = dict(color=palette.OPTION1),
          name = "magnitude option 1",
          xaxis = 'x2',
          yaxis = 'y2',
      ))
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = magOpt2,
          mode = 'lines',
          line = dict(color=palette.OPTION2),
          name = "magnitude option 2",
          xaxis = 'x2',
          yaxis = 'y2',
      ))

  if df:

      # plot the utility of option 1 and option 2
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility1,
              mode = 'lines',
              line = dict(color=palette.OPTION1),
              name = "utility option 1",
              xaxis = 'x3',
              yaxis = 'y3',
          ))
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility2,
              mode = 'lines',
              line = dict(color=palette.OPTION2),
              name = "utility option 2",
              xaxis = 'x3',
              yaxis = 'y3',
          ))

      # label the axes
      fig.update_layout(xaxis3_title="trial number", yaxis3_title="utility")

      fig.update_xaxes(range=[0, nTrials], row=3, col=1)

  if df:
    # label the axes
    fig.update_layout(xaxis2_title="", yaxis2_title="magnitude")
  else:
    # label the axes
    fig.update_layout(xaxis2_title="trial number", yaxis2_title="magnitude")

  # set x-axis range for all subplots
  # this sets the range for all x-axes
  fig.update_xaxes(range=[0, nTrials])
  # this is needed to set the range for the second subplot
  fig.update_xaxes(range=[0, nTrials], row=2, col=1)

  return fig

def visualise_alpha_distributions(stableAlphas, volatileAlphas, title):
  fig = px.histogram(pd.DataFrame({"stable block": stableAlphas,
                                  "volatile block": volatileAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay",
                                  color_discrete_map={"stable block": palette.STABLE,
                                                      "volatile block": palette.VOLATILE})

  fig.update_layout(title= title, xaxis_title="learning rate", legend_title_text="")
  fig.update_xaxes(range=[0, 1])
  
  fig.update_traces(xbins=dict(
        start=0.0,
        end=1.0,
        size=0.05
    ), 
    row = 1
    )

  fig.show()
  
  
def visualise_alpha_difference(stableAlphas, volatileAlphas, title):
  fig = px.histogram(pd.DataFrame({"volatile α − stable α": volatileAlphas - stableAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay",
                                  color_discrete_sequence=[palette.NEUTRAL])

  fig.update_layout(title= title, xaxis_title="volatile α − stable α", legend_title_text="", showlegend=False)
  fig.update_xaxes(range=[-1, 1])
  
  fig.update_traces(xbins=dict(
        start=-1.0,
        end=1.0,
        size=0.1
    ), 
    row = 1
    )

  fig.show()
  
def plot_parameter_corrs(df):
  if any(df.columns == 'alphaStable'):
    if any(df.columns == 'omega'):
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='α stable',
                                    values=df.alphaStable),
                                dict(label='α volatile',
                                    values=df.alphaVolatile),
                                dict(label='β',
                                    values=df.beta),
                                dict(label='ω',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    marker = dict(color=palette.NEUTRAL, size=6),
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='α stable',
                                    values=df.alphaStable),
                                dict(label='α volatile',
                                    values=df.alphaVolatile),
                                dict(label='β',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    marker = dict(color=palette.NEUTRAL, size=6),
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
  else:
    if any(df.columns == 'omega'):
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='α',
                                    values=df.alpha),
                                dict(label='β',
                                    values=df.beta),
                                dict(label='ω',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    marker = dict(color=palette.NEUTRAL, size=6),
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='α',
                                    values=df.alpha),
                                dict(label='β',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    marker = dict(color=palette.NEUTRAL, size=6),
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    
  fig.show()


def plot_model_frequencies(bestModels, sd=None, exceedance_probabilities=None, modelNames=['1 α, multiplicative utility', '2 α, multiplicative utility', '1 α, additive utility', '2 α, additive utility']):
    # use plotly to plot the model frequencies
    frequencies = bestModels/np.sum(bestModels)
    if sd is not None:
        # plot with error bars using go.Bar
        fig = go.Figure(data=go.Bar(
            x=modelNames,
            y=frequencies,
            marker_color=palette.NEUTRAL,
            error_y=dict(type='data', array=sd, color=palette.TRUTH)
        ))
    else:
        fig = go.Figure(data=go.Bar(
            x=modelNames,
            y=frequencies,
            marker_color=palette.NEUTRAL
        ))
    if exceedance_probabilities is not None:
        text_labels = [f'exceedance probability {val:.3f}' for val in exceedance_probabilities]
        y_positions = frequencies + (sd + 0.05 if sd is not None else 0.05)
        fig.add_trace(go.Scatter(
            x=modelNames,
            y=y_positions,
            mode='text',
            text=text_labels,
            textfont=dict(color=palette.TRUTH)
        ))
        
    fig.update_layout(yaxis_title='frequency', xaxis_title='', showlegend=False)
    fig.show()
