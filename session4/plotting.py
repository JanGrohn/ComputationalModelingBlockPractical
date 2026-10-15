# numpy is a library used to do all kinds of mathematical operations
import numpy as np

# pandas allows us to organise data as tables (called "dataframes")
import pandas as pd

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# this allows us to calculate trendlines
from sklearn.linear_model import LinearRegression

# set the style of the plotly figures
pio.templates.default = "none"

# import some custom functions we wrote
from session4 import fitting, loading


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
          line = dict(color='black', dash='dash'),
          name = "true probability"
      ))

  # check if we should plot probOpt1, and if so plot it as a line
  if df:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = probOpt1,
            mode = 'lines',
            line = dict(color='red'),
            name = "RL model probability"
        ))

  # check if we should plot choiceProb1, and if so plot it as a scatterplot
  if df:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = choiceProb1,
            mode = 'markers',
            marker = dict(size = 5, symbol='x', color='blue'),
            name = "choice probability"
        ))

  # get correct and incorrect choices
  correct_choices = np.array([choice1[i] if (opt1Rewarded[i] and choice1[i]) or  (not opt1Rewarded[i] and not choice1[i]) else np.nan for i in range(nTrials)])
  incorrect_choices = np.array([choice1[i] if (opt1Rewarded[i] and not choice1[i]) or  (not opt1Rewarded[i] and choice1[i]) else np.nan for i in range(nTrials)])

  # plot choices as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = correct_choices,
          mode = 'markers',
          marker = dict(size = 5, opacity=0.8, color="green"),
          name = "correct choice"
      ))

  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = incorrect_choices,
          mode = 'markers',
          marker = dict(size = 5, opacity=0.8, color="red"),
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
          line = dict(color='orange'),
          name = "magnitude option 1",
          xaxis = 'x2',
          yaxis = 'y2',
      ))
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = magOpt2,
          mode = 'lines',
          line = dict(color='green'),
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
              line = dict(color='orange'),
              name = "utility option 1",
              xaxis = 'x3',
              yaxis = 'y3',
          ))
      fig.add_trace(
          go.Scatter(
              x = list(range(nTrials)),
              y = utility2,
              mode = 'lines',
              line = dict(color='green'),
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
                                  color="variable", x="value", marginal="box", barmode="overlay")

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
  fig = px.histogram(pd.DataFrame({"volatile alpha - stable alpha": volatileAlphas - stableAlphas}).melt(),
                                  color="variable", x="value", marginal="box", barmode="overlay")

  fig.update_layout(title= title, xaxis_title="volatile alpha - stable alpha", legend_title_text="", showlegend=False)
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
                    dimensions=[dict(label='alpha stable',
                                    values=df.alphaStable),
                                dict(label='alpha volatile',
                                    values=df.alphaVolatile),
                                dict(label='beta',
                                    values=df.beta),
                                dict(label='omega',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha stable',
                                    values=df.alphaStable),
                                dict(label='alpha volatile',
                                    values=df.alphaVolatile),
                                dict(label='beta',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
  else:
    if any(df.columns == 'omega'):
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha',
                                    values=df.alpha),
                                dict(label='beta',
                                    values=df.beta),
                                dict(label='omega',
                                    values=df['omega'])],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    else:
      fig = go.Figure(data=go.Splom(
                    dimensions=[dict(label='alpha',
                                    values=df.alpha),
                                dict(label='beta',
                                    values=df.beta)],
                    showupperhalf=False,
                    diagonal_visible=False, 
                    hovertemplate = 'participant ID: %{customdata}',
                    customdata = df.ID
                    ))
    
  fig.show()

def plot_recovered_parameters(recoveryData):
  '''
  Plots the simulated against the recovered parameters

    Parameters:
        recoveryData(DataFrame): DataFrame with columns simulatedAlpha,
          simulatedBeta, recoveredAlpha, recoveredBeta
  '''

  if any(recoveryData.columns == 'alphaStable'):
      if any(recoveryData.columns == 'omega'):
          fig = make_subplots(rows=1, cols=4,
                      subplot_titles=("alpha stable","alpha volatile","omega","beta"))
          # Add line traces
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=1)      
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=2)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=3)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=4)

          # Add trendlines
          model = LinearRegression()
          model.fit(recoveryData[["alphaStable"]].values.reshape(-1,1), recoveryData[["recovered2AddAlphaS"]]) 
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=1)
          
          model = LinearRegression()
          model.fit(recoveryData[["alphaVolatile"]].values.reshape(-1,1), recoveryData[["recovered2AddAlphaV"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=2)
          
          model = LinearRegression()
          model.fit(recoveryData[["omega"]].values.reshape(-1,1), recoveryData[["recovered2AddOmega"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=3)
          
          model = LinearRegression()
          model.fit(recoveryData[["beta"]].values.reshape(-1,1), recoveryData[["recovered2AddBeta"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=4)
          
          # Add scatter traces
          fig.add_trace(go.Scatter(
                            x=recoveryData["alphaStable"],
                            y=recoveryData["recovered2AddAlphaS"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha volatile: %{customdata[2]}<br>Recovered alpha volatile: %{customdata[3]}<br>Simulated omega: %{customdata[4]}<br>Recovered omega: %{customdata[5]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered2AddBeta"],3), round(recoveryData["alphaVolatile"],3), round(recoveryData["recovered2AddAlphaV"],3), round(recoveryData["omega"],3), round(recoveryData["recovered2AddOmega"],3)), axis=-1),
                            ),
                    row=1, col=1)
          
          fig.add_trace(go.Scatter(
                            x=recoveryData["alphaVolatile"],
                            y=recoveryData["recovered2AddAlphaV"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha stable: %{customdata[2]}<br>Recovered alpha stable: %{customdata[3]}<br>Simulated omega: %{customdata[4]}<br>Recovered omega: %{customdata[5]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered2AddBeta"],3), round(recoveryData["alphaStable"],3), round(recoveryData["recovered2AddAlphaS"],3), round(recoveryData["omega"],3), round(recoveryData["recovered2AddOmega"],3)), axis=-1),
                            ),
                    row=1, col=2)
          
          fig.add_trace(go.Scatter(
                            x=recoveryData["omega"],
                            y=recoveryData["recovered2AddOmega"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha stable: %{customdata[2]}<br>Recovered alpha stable: %{customdata[3]}<br>Simulated alpha volatile: %{customdata[4]}<br>Recovered alpha volatile: %{customdata[5]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered2AddBeta"],3), round(recoveryData["alphaStable"],3), round(recoveryData["recovered2AddAlphaS"],3), round(recoveryData["alphaVolatile"],3), round(recoveryData["recovered2AddAlphaV"],3)), axis=-1),
                            ),
                    row=1, col=3)

          fig.add_trace(go.Scatter(
                            x=recoveryData["beta"],
                            y=recoveryData["recovered2AddBeta"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated alpha stable: %{customdata[0]}<br>Recovered alpha stable: %{customdata[1]}<br>Simulated alpha volatile: %{customdata[2]}<br>Recovered alpha volatile: %{customdata[3]}<br>Simulated omega: %{customdata[4]}<br>Recovered omega: %{customdata[5]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["alphaStable"],3), round(recoveryData["recovered2AddAlphaS"],3), round(recoveryData["alphaVolatile"],3), round(recoveryData["recovered2AddAlphaV"],3), round(recoveryData["omega"],3), round(recoveryData["recovered2AddOmega"],3)), axis=-1),
                            ),
                    row=1, col=4)

          fig.update_layout(xaxis1_title="simulated", yaxis1_title="recovered", showlegend=False)
          fig.update_layout(xaxis2_title="simulated", yaxis2_title="recovered", showlegend=False)
          fig.update_layout(xaxis3_title="simulated", yaxis3_title="recovered", showlegend=False)
          fig.update_layout(xaxis4_title="simulated", yaxis4_title="recovered", showlegend=False)
      else:
          fig = make_subplots(rows=1, cols=3,
                      subplot_titles=("alpha stable","alpha volatile","beta"))
          
          # Add line traces
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=1)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=2)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=3)
          
           # Add trendlines
          model = LinearRegression()
          model.fit(recoveryData[["alphaStable"]].values.reshape(-1,1), recoveryData[["recovered2MulAlphaS"]]) 
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=1)
          
          model = LinearRegression()
          model.fit(recoveryData[["alphaVolatile"]].values.reshape(-1,1), recoveryData[["recovered2MulAlphaV"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=2)
          
          model = LinearRegression()
          model.fit(recoveryData[["beta"]].values.reshape(-1,1), recoveryData[["recovered2MulBeta"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=3)
            
          # Add scatter traces
          fig.add_trace(go.Scatter(
                            x=recoveryData["alphaStable"],
                            y=recoveryData["recovered2MulAlphaS"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate='Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha volatile: %{customdata[2]}<br>Recovered alpha volatile: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered2MulBeta"],3), round(recoveryData["alphaVolatile"],3), round(recoveryData["recovered2MulAlphaV"],3)), axis=-1),
                            ),
                    row=1, col=1)
          
          fig.add_trace(go.Scatter(
                            x=recoveryData["alphaVolatile"],
                            y=recoveryData["recovered2MulAlphaV"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha stable: %{customdata[2]}<br>Recovered alpha stable: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered2MulBeta"],3), round(recoveryData["alphaStable"],3), round(recoveryData["recovered2MulAlphaS"],3)), axis=-1),
                            ),
                    row=1, col=2)

          fig.add_trace(go.Scatter(
                            x=recoveryData["beta"],
                            y=recoveryData["recovered2MulBeta"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated alpha stable: %{customdata[0]}<br>Recovered alpha stable: %{customdata[1]}<br>Simulated alpha volatile: %{customdata[2]}<br>Recovered alpha volatile: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["alphaStable"],3), round(recoveryData["recovered2MulAlphaS"],3), round(recoveryData["alphaVolatile"],3), round(recoveryData["recovered2MulAlphaV"],3)), axis=-1),
                            ),
                    row=1, col=3)

          fig.update_layout(xaxis1_title="simulated", yaxis1_title="recovered", showlegend=False)
          fig.update_layout(xaxis2_title="simulated", yaxis2_title="recovered", showlegend=False)
          fig.update_layout(xaxis3_title="simulated", yaxis3_title="recovered", showlegend=False)

  else:
      if any(recoveryData.columns == 'omega'):
          fig = make_subplots(rows=1, cols=3,
                      subplot_titles=("alpha","omega","beta"))
          
          # Add line traces
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=1)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=2)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=3)
          
          # Add trendlines
          model = LinearRegression()
          model.fit(recoveryData[["alpha"]].values.reshape(-1,1), recoveryData[["recovered1AddAlpha"]]) 
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=1)
          
          model = LinearRegression()
          model.fit(recoveryData[["omega"]].values.reshape(-1,1), recoveryData[["recovered1AddOmega"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=2)
          
          model = LinearRegression()
          model.fit(recoveryData[["beta"]].values.reshape(-1,1), recoveryData[["recovered1AddBeta"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=3)
            
          # Add scatter traces
          fig.add_trace(go.Scatter(
                            x=recoveryData["alpha"],
                            y=recoveryData["recovered1AddAlpha"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated omega: %{customdata[2]}<br>Recovered omega: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered1AddBeta"],3), round(recoveryData["omega"],3), round(recoveryData["recovered1AddOmega"],3)), axis=-1),
                            ),
                    row=1, col=1)
          
          fig.add_trace(go.Scatter(
                            x=recoveryData["omega"],
                            y=recoveryData["recovered1AddOmega"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<br>Simulated alpha: %{customdata[2]}<br>Recovered alpha: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered1AddBeta"],3), round(recoveryData["alpha"],3), round(recoveryData["recovered1AddAlpha"],3)), axis=-1),
                            ),
                    row=1, col=2)

          fig.add_trace(go.Scatter(
                            x=recoveryData["beta"],
                            y=recoveryData["recovered1AddBeta"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated alpha: %{customdata[0]}<br>Recovered alpha: %{customdata[1]}<br>Simulated omega: %{customdata[2]}<br>Recovered omega: %{customdata[3]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["alpha"],3), round(recoveryData["recovered1AddAlpha"],3), round(recoveryData["omega"],3), round(recoveryData["recovered1AddOmega"],3)), axis=-1),
                            ),
                    row=1, col=3)

          fig.update_layout(xaxis1_title="simulated", yaxis1_title="recovered", showlegend=False)
          fig.update_layout(xaxis2_title="simulated", yaxis2_title="recovered", showlegend=False)
          fig.update_layout(xaxis3_title="simulated", yaxis3_title="recovered", showlegend=False)
      else:
          fig = make_subplots(rows=1, cols=2,
                      subplot_titles=("alpha","beta"))
          
          # Add line traces
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=1)
          fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='black', width=1)), row=1, col=2)

          # Add trendlines
          model = LinearRegression()
          model.fit(recoveryData[["alpha"]].values.reshape(-1,1), recoveryData[["recovered1MulAlpha"]]) 
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=1)
          
          model = LinearRegression()
          model.fit(recoveryData[["beta"]].values.reshape(-1,1), recoveryData[["recovered1MulBeta"]])
          x_range = np.linspace(0, 1, 100)
          y_range = model.predict(x_range.reshape(-1, 1))
          fig.add_trace(go.Scatter(x=x_range, y=y_range[:,0], mode='lines', line=dict(color='red', width=2)), row=1, col=2)
          
          # Add scatter traces
          fig.add_trace(go.Scatter(
                            x=recoveryData["alpha"],
                            y=recoveryData["recovered1MulAlpha"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated beta: %{customdata[0]}<br>Recovered beta: %{customdata[1]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["beta"],3), round(recoveryData["recovered1MulBeta"],3)), axis=-1),
                            ),
                    row=1, col=1)

          fig.add_trace(go.Scatter(
                            x=recoveryData["beta"],
                            y=recoveryData["recovered1MulBeta"],
                            mode='markers',
                            marker=dict(color='rgba(0, 0, 255, 0.1)'),
                            hovertemplate = 'Simulated alpha: %{customdata[0]}<br>Recovered alpha: %{customdata[1]}<extra></extra>',
                            customdata = np.stack((round(recoveryData["alpha"],3), round(recoveryData["recovered1MulAlpha"],3)), axis=-1),
                            ),
                    row=1, col=2)

          fig.update_layout(xaxis1_title="simulated", yaxis1_title="recovered", showlegend=False)
          fig.update_layout(xaxis2_title="simulated", yaxis2_title="recovered", showlegend=False)

  # Show the figure
  fig.show()
  
def visualise_LR_recovery(recov1AlphaMul, recov2AlphaMul, recov1AlphaAdd, recov2AlphaAdd, p = 0.05, degrees_of_freedom = 75):
  models=['multiplicative utility', 'additive utility']
  nReps = int((max(recov2AlphaAdd.ID)+1)/degrees_of_freedom)
  p_recov1AlphaMul, _ = fitting.recov_chi2_test(recov1AlphaMul, degrees_of_freedom = degrees_of_freedom, nReps = nReps)
  p_recov2AlphaMul, _ = fitting.recov_chi2_test(recov2AlphaMul, degrees_of_freedom = degrees_of_freedom, nReps = nReps)
  _, p_recov1AlphaAdd = fitting.recov_chi2_test(recov1AlphaAdd, degrees_of_freedom = degrees_of_freedom, nReps = nReps)
  _, p_recov2AlphaAdd = fitting.recov_chi2_test(recov2AlphaAdd, degrees_of_freedom = degrees_of_freedom, nReps = nReps)
  

  fig = go.Figure(data=[
      go.Bar(name='1 alpha', x=models, y=[sum(p_recov1AlphaMul<p)/nReps, sum(p_recov1AlphaAdd<p)/nReps]),
      go.Bar(name='2 alphas', x=models, y=[sum(p_recov2AlphaMul<p)/nReps, sum(p_recov2AlphaAdd<p)/nReps])
  ])

  fig.update_layout(barmode='group', title = 'significant LR tests at p < ' + str(p), xaxis_title = 'simulated with', yaxis_title = 'proportion significant LR tests')
  fig.show()
  
def visualise_BIC_recovery(recov1AlphaMul, recov2AlphaMul, recov1AlphaAdd, recov2AlphaAdd, num_simulated_participants = 75):
  nReps = int((max(recov2AlphaAdd.ID)+1)/num_simulated_participants)
  BICs = np.vstack((
      fitting.recov_BICs(recov1AlphaMul, num_simulated_participants = num_simulated_participants, nReps = nReps),
      fitting.recov_BICs(recov2AlphaMul, num_simulated_participants = num_simulated_participants, nReps = nReps),
      fitting.recov_BICs(recov1AlphaAdd, num_simulated_participants = num_simulated_participants, nReps = nReps),
      fitting.recov_BICs(recov2AlphaAdd, num_simulated_participants = num_simulated_participants, nReps = nReps)))/nReps

  labels = ['1 alpha, multiplicative utility','2 alphas, multiplicative utility', '1 alpha, additive utility', '2 alphas, additive utility']
  heat = go.Heatmap(z=BICs,
                    x=labels,
                    y=labels,
                    xgap=1, ygap=1,
                    colorbar_thickness=20,
                    colorbar_ticklen=3)

  layout = go.Layout(title_text='Proportion of BIC comparisons that identify a model', 
                    width=600, height=600,
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                    yaxis_autorange='reversed')
    
  fig=go.Figure(data=[heat], layout=layout)
  fig.update_layout(
      margin=dict(l=230, r=30, t=30, b=230),
      xaxis_title="recovered with",
      yaxis_title="simulated with"
  ) 
  fig.update_xaxes(tickangle=-45)       
  fig.show() 
  
def visualise_t_test_recovery(recov1AlphaMul, recov2AlphaMul, recov1AlphaAdd, recov2AlphaAdd, p = 0.05, num_simulated_participants = 75):
  models=['multiplicative utility', 'additive utility']
  nReps = int((max(recov2AlphaAdd.ID)+1)/num_simulated_participants)
  p_recov1AlphaMul, _ = fitting.recov_t_test(recov1AlphaMul, num_simulated_participants = num_simulated_participants, nReps = nReps)
  p_recov2AlphaMul, _ = fitting.recov_t_test(recov2AlphaMul, num_simulated_participants = num_simulated_participants, nReps = nReps)
  _, p_recov1AlphaAdd = fitting.recov_t_test(recov1AlphaAdd, num_simulated_participants = num_simulated_participants, nReps = nReps)
  _, p_recov2AlphaAdd = fitting.recov_t_test(recov2AlphaAdd, num_simulated_participants = num_simulated_participants, nReps = nReps)
  

  fig = go.Figure(data=[
      go.Bar(name='1 alpha', x=models, y=[sum(p_recov1AlphaMul<p)/nReps, sum(p_recov1AlphaAdd<p)/nReps]),
      go.Bar(name='2 alphas', x=models, y=[sum(p_recov2AlphaMul<p)/nReps, sum(p_recov2AlphaAdd<p)/nReps])
  ])

  fig.update_layout(barmode='group', title = 'significant t tests at p < ' + str(p), xaxis_title = 'simulated with', yaxis_title = 'proportion significant t tests')
  fig.show()