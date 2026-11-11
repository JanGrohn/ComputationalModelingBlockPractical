# json reads the data file the task in session 1 lets you download
import json

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

# the task's condition 1 is the stable schedule and condition 2 the volatile one
CONDITIONS = {1: "stable", 2: "volatile"}

def load_own_data(paths):
  '''
  Returns your own data from the task you played in session 1.

    Parameters:
        paths(list of str): the participantData.json files the task let you
          download at the end, one for each condition you played.

    Returns:
        blocks(dict): one entry, "stable" or "volatile", for each condition you
          played. Each holds the arrays choice1, magOpt1, magOpt2 and
          opt1Rewarded, as described in load_blain.
  '''

  blocks = {}
  for path in paths:
    with open(path) as file:
      events = json.load(file)

    # the file is a list of events: who you are, the condition you picked, and
    # one "Select Left" or "Select Right" event for every choice you made
    conditions = [event["condition"] for event in events if "condition" in event]
    choices    = [event for event in events if event.get("type") in ("Select Left", "Select Right")]
    if len(conditions) != 1 or conditions[0] not in CONDITIONS or not choices:
      raise ValueError(
        f"{path} does not look like a file downloaded from the task at the end of "
        f"session 1: it should name one condition (1 or 2) and contain your choices."
      )

    condition = CONDITIONS[conditions[0]]
    if condition in blocks:
      raise ValueError(
        f"Two of your files are from the {condition} condition. Upload one file "
        f"per condition you played."
      )

    # displaySide is 1 when option 1 (green) was shown on the left, so you chose
    # option 1 whenever you picked the side it was on
    choice1      = np.array([(event["type"] == "Select Left") == (event["displaySide"] == 1) for event in choices], dtype = int)
    magOpt1      = np.array([event["reward1"] for event in choices], dtype = int)
    magOpt2      = np.array([event["reward2"] for event in choices], dtype = int)
    opt1Rewarded = np.array([event["oneRewarded"] for event in choices], dtype = int)

    blocks[condition] = (choice1, magOpt1, magOpt2, opt1Rewarded)

  return blocks