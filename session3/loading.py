# numpy is a library used to do all kinds of mathematical operations
import numpy as np

# load in the dataset
from scipy.io import loadmat

DATA_FILE = "Blain_MoodTracksLearning_data.mat"

try:
  blain_data = loadmat(DATA_FILE)
except FileNotFoundError:
  raise FileNotFoundError(
    f"Could not find {DATA_FILE} in the current working directory. Run the "
    f"cell at the top of the notebook that downloads it, or start the notebook "
    f"from the folder that contains it."
  ) from None

def load_blain(ID, data = blain_data):
  '''
  Returns one participant's data from the Blain and Rutledge (2020) dataset.

    Parameters:
        ID(int): participant number, 0 to 74. Participants 0 to 36 did the
          stable block first, participants 37 to 74 the volatile block first.
        data(dict): the loaded .mat file, defaults to the dataset loaded above.

    Returns:
        trueProbability(float array): the probability with which option 1 is
          rewarded on each trial
        choice1(int array): 1 if option 1 was chosen on a trial, 0 otherwise
        magOpt1(int array): reward points on offer for option 1 on each trial
        magOpt2(int array): reward points on offer for option 2 on each trial
        opt1Rewarded(int array): 1 if option 1 was rewarded on a trial, 0 if
          option 2 was rewarded
  '''

  if ID < 37:
    order = "stable2volatile_data"
  else:
    order = "volatile2stable_data"
    ID -= 37

  trueProbability = data[order]["main"][0][ID]["task_mu"][0][0][0]
  outcome = data[order]["main"][0][ID]["maintask"][0][0][:,14] > 0
  numTrials = len(outcome)
  choice1       = np.empty(numTrials, dtype = int)
  magOpt1       = np.empty(numTrials, dtype = int)
  magOpt2       = np.empty(numTrials, dtype = int)
  opt1Rewarded  = np.empty(numTrials, dtype = int)
  
  for i in range(numTrials):
    choice1[i]  = 2 - data[order]["main"][0][ID]["list_pair"][0][0][int(data[order]["main"][0][ID]["maintask"][0][0][i,4]-1),i]
    magOpt1[i]  = data[order]["main"][0][ID]["mat_mag"][0][0][data[order]["main"][0][ID]["index_mag"][0][0][:,i][0]-1,0]
    magOpt2[i]  = data[order]["main"][0][ID]["mat_mag"][0][0][data[order]["main"][0][ID]["index_mag"][0][0][:,i][0]-1,1]
    if choice1[i]:
      if outcome[i]:
        opt1Rewarded[i] = 1
      else:
        opt1Rewarded[i] = 0
    else:
      if outcome[i]:
        opt1Rewarded[i] = 0
      else:
        opt1Rewarded[i] = 1

  return trueProbability, choice1, magOpt1, magOpt2, opt1Rewarded