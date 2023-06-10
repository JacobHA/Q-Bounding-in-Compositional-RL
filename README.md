Thank you for taking the time to visit the Supplementary Materials for the paper "Bounding the Optimal Value Function in Compositional Reinforcement Learning".

In this folder you will find the Code used to generate the results in the paper, as well as the data ("export" folder) used to generate the figures in the paper.

The optimal hyperparameters (as stated in the Appendix) for each clip method are in the dqn_sweep.py file.

## Reconstructing learning curves

1. Reconstruct the reward plot the results with `python plot.py`
2. Reconstruct the bound violation plot the results with `python plot_BV.py`

## Regenerate the data

You'll need a Weights and Biases account to run sweeps.

1. Pretrain the DQN models on primitive tasks with `python pretrain.py`
2. Run a sweep with optimal hyperparameters with `python dqn_sweep.py`
3. Export the results with `python export.py --entity <your wandb username>`


## Individual Experiments

Following can be tested and viewed through tensorboard locally:

Start tensorboard with `tensorboard --logdir=tmp` from the Code/ dir.
View the dashboard with browser at `localhost:6006`


### Baseline DQN

train SB3 DQN on custom maze `6x6L_AND_D` in 200k steps 

`python dqn_baseline.py`


### Composed DQN

train SB3 DQN on custom maze `6x6L_AND_D` in 200k steps. Requires two
pretrained models, `6x6L` and `6x6D` in `models/` dir. Run `python pretrain.py` to generate the two models.

`python dqn_composed.py --env 6x6L_AND_D --clipmethod none --comptype and`

### Tabular Experiments (Supplementary Material)

[TabularExperiments/Videos/maxent_4room_sweep.gif](TabularExperiments/Videos/maxent_4room_sweep.gif)

[TabularExperiments/Videos/std_4room_sweep.gif](TabularExperiments/Videos/std_4room_sweep.gif)

In the TabularExperiments folder, there is a frozen_lake_tests.py file with options in the `main` function that can be used to run the tabular experiments. 


