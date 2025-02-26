# numpy is a libarary used to do all kinds of mathematical operations
import numpy as np

# we use plotly to make our figures
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px

# set the style of the plotly figures
pio.templates.default = "none"

# this allows us to make interactive figures
import ipywidgets as widgets

# Add type hint imports
from typing import Optional, Callable, Union
import numpy.typing as npt
from plotly.graph_objs._figure import Figure
from ipywidgets import VBox

def plot_schedule(
    opt1Rewarded:    npt.NDArray,
    trueProbability: npt.NDArray,
    probOpt1:        Optional[npt.NDArray] = None
    ) -> Figure:
  '''
  Plots the experimental schedule and the RL model estimate.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        probOpt1(float array): how likely option 1 is rewarded on each trial
          according to the RL model. Defaults to None, which excludes it from
          the plot.
    
    Returns:
        Figure: A plotly figure object
  '''

  # compute number of trials
  nTrials = len(opt1Rewarded)

  # initalise figure
  fig = go.Figure()

  # plot opt1rewarded as a scatterplot
  fig.add_trace(
      go.Scatter(
          x = list(range(nTrials)),
          y = opt1Rewarded,
          mode = 'markers',
          marker = dict(color="blue", size = 5),
          name = "trial outcomes"
      ))

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
  if probOpt1 is not None:
    fig.add_trace(
        go.Scatter(
            x = list(range(nTrials)),
            y = probOpt1,
            mode = 'lines',
            line = dict(color='red'),
            name = "RL model probability"
        ))

  # label the axes
  fig.update_layout(xaxis_title="trial number", yaxis_title="green rewarded?")

  # set the range for the axes
  fig.update_xaxes(range=[0, nTrials])
  fig.update_yaxes(range=[-0.05, 1.05])

  return fig

def plot_interactive_RL_model(
    opt1Rewarded:      npt.NDArray,
    trueProbability:   npt.NDArray,
    RL_model:          Callable,
    generate_schedule: Callable,
    change_trueProb:   bool = True,
    ) -> None:
  '''
  Plots the experimental schedule and the RL model estimate with sliders to
  change the model parameters.

    Parameters:
        opt1rewarded(int array): 1 if option 1 is rewarded on a trial, 0 if
          option 2 is rewarded on a trial.
        trueProbability(float array): The probability with which option 1 is
          rewareded on each trial.
        RL_model(function): the RL model function to use to generate probOpt1.
        generate_schedule(function): Function to generate new schedules
        change_trueProb(bool): whether to include a slider to change the true
          probability. Defaults to True.
  '''

  # compute number of trials
  nTrials = len(opt1Rewarded)

  # this initiates some sliders that can be used to dynamically change parameter values
  alphaSlider = widgets.FloatSlider(
                                  value=0.1,
                                  max=1,
                                  min=0.001,
                                  step=0.01,
                                  description='alpha:',
                                  continuous_update=False
                                  )

  startingProbSlider = widgets.FloatSlider(
                                  value=0.5,
                                  max=1,
                                  min=0,
                                  step =0.01,
                                  description='startingProb:',
                                  continuous_update=False
                                  )
  
  trueProbSlider = widgets.FloatSlider(
                                  value=0.8,
                                  max=1,
                                  description='trueProb:',
                                  continuous_update=False
                                  )
  if change_trueProb:
    sliders = widgets.VBox(children=[alphaSlider,
                                    startingProbSlider,
                                    trueProbSlider])
  else:
    sliders = widgets.VBox(children=[alphaSlider,
                                    startingProbSlider])
  
  probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)

  # call the figure function we wrote and make it interactive
  fig = go.FigureWidget(plot_schedule(opt1Rewarded,trueProbability,probOpt1))

  def change_model(change: dict) -> None:
    # get current values
    opt1Rewarded = fig.data[0].y
    trueProbability = trueProbSlider.value
    # rerun the RL model
    probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)
    # update the figure
    with fig.batch_update():
      fig.data[2].y = probOpt1

  def change_schedule(change: dict) -> None:
    # generate a new schedule
    trueProbability = np.ones(nTrials, dtype = float)*trueProbSlider.value
    opt1Rewarded = generate_schedule(trueProbability)
    # rerun the RL model
    probOpt1 = RL_model(opt1Rewarded,alphaSlider.value,startingProbSlider.value)
    # update the figure
    with fig.batch_update():
      fig.data[0].y = opt1Rewarded
      fig.data[1].y = trueProbability
      fig.data[2].y = probOpt1

  # run the functions if a slider value changes
  alphaSlider.observe(change_model, names="value")
  startingProbSlider.observe(change_model, names="value")
  if change_trueProb:
    trueProbSlider.observe(change_schedule, names="value")

  # show the figure and the sliders
  display(widgets.VBox([sliders, fig]))

def visualise_utility_function(
    utility_function: Callable,
    omega:            bool = False,
    nSamples:         int = 100
    ) -> None:
  '''
  Visualises a utility function using plotly.

    Parameters:
        utility_function(function): The utility function to visualise.
        omega(bool): Whether the utility function has an omega parameter.
        nSamples(int): The number of samples to take from the utility function
          to visualise it.
  '''

  # get a slider for omega
  omegaSlider = widgets.FloatSlider(
                            value=0.5,
                            max=1,
                            min=0,
                            step=0.01,
                            description='omega:',
                            continuous_update=False)

  # the range of mag and prob values for which we want to visualise the function
  magRange  = np.linspace(1, 100, nSamples)
  probRange = np.linspace(0, 1, nSamples)

  def compute_utilityMatrix(
      utility_function: Callable = utility_function,
      show_omega: bool = omega,
      nSamples: int = nSamples,
      omega: float = omegaSlider.value
      ) -> npt.NDArray:

    # empty matrix to fill in with utility values
    utilityMatrix = np.zeros((nSamples,nSamples))

    # loop through all possible values of mag and prob and get their utility
    for mag in range(nSamples):
      for prob in range(nSamples):
        if show_omega:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob], omega)
        else:
          utilityMatrix[mag,prob] = utility_function(magRange[mag], probRange[prob])
    
    return utilityMatrix
  
  utilityMatrix = compute_utilityMatrix(utility_function)

  # make a 3d plot of the utility function
  fig = go.Figure(go.Surface(z = utilityMatrix,
                            y = magRange,
                            x = probRange))

  fig.update_scenes(yaxis_title='magnitude',
                    xaxis_title='probability',
                    zaxis_title='utility')

  fig.update_layout(scene_camera=dict(eye=dict(x=-2, y=-2, z=2)),
                    autosize=False,
                    width=500,
                    height=500,
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0,
                        pad=4
                    ))

  # if omega is true, add a slider to change omega
  if omega:
    # make the figure interactive
    fig = go.FigureWidget(fig)

    def change_omega(change: dict) -> None:
      # calculate the utility matrix for the new omega value
      utilityMatrix = compute_utilityMatrix(omega = omegaSlider.value)
      # update the figure
      with fig.batch_update():
        fig.data[0].z = utilityMatrix
    
    # listen to changes of the omega slider
    omegaSlider.observe(change_omega, names="value")

    # show the slider and figure
    display(widgets.VBox([omegaSlider, fig]))

  else:
     display(fig)
     
def plot_RL_weights(T: int = 10) -> None:
  '''
  Plots the RL weights values of alpha.
  
      Parameters:
          T(int): the number of trials back to plot the weights for.
  '''
  
  # get a slider for alpha
  alphaSlider = widgets.FloatSlider(
                              value=0.1,
                              max=1,
                              min=0.01,
                              step=0.01,
                              description='alpha:',
                              continuous_update=False)

  def compute_weights(alpha: float) -> npt.NDArray:
    # empty weight vector to assign into
    weight = np.empty(T, dtype = float)
    # calculate the weight on every trial back
    for t in range(T):
        # EXERCISE: explain/derive the following equation:
        weight[t] = alpha*(1-alpha)**(T-(t+1));
    return weight

  # start a new figure
  fig = go.FigureWidget(go.Scatter(
            x = np.arange(-T, 0, 1),
            y = compute_weights(alphaSlider.value),
            mode = 'lines',
            line = dict(color='black')
        ))

  # label the axes
  fig.update_layout(xaxis_title="delay (trials)", yaxis_title="weight")

  def change_alpha(change: dict) -> None:
    # get the current value of alpha
    alpha = alphaSlider.value
    # calculate the corresponding weights
    weight = compute_weights(alpha)
    # update the figure
    with fig.batch_update():
      fig.data[0].y = weight

  # listen to changes of the alpha slider
  alphaSlider.observe(change_alpha, names="value")

  # show the slider and figure
  display(widgets.VBox([alphaSlider, fig]))
