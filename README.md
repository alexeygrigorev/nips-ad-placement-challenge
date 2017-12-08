# Ad Placement Challenge

The winning solution to the [Ad Placement Challenge](https://www.crowdai.org/challenges/nips-17-workshop-criteo-ad-placement-challenge) by Criteo for the 
[NIPS'17 Causal Inference and Machine Learning Workshop](https://sites.google.com/view/causalnips2017)

The solution is simple:

* Take the data as is, train a FTRL model (using [libftrl-python](https://github.com/alexeygrigorev/libftrl-python))
* Post-process the predictions:
  * Apply the sigmoid fuction to the predictions and scale the result by some constant value
  * For each group add +15 to the max value

Report:

* The video: https://www.youtube.com/watch?v=d5I0baV51vk
* The report: https://arxiv.org/abs/1712.01913 (also in `report/`)

Environment:

* ubuntu 14.04
* 8 cores, 32 gb of ram
* anaconda with python 3.5.2
* additional packages not from anaconda: `tqdm` and `ftrl`

Running the solution

* `mkdir tmp`
* `python 01-cv-split.py` # takes ~22 mins
* `python 02-prepare-data.py` # ~17 mins
* `python 03-train-model.py` # ~25 mins
* `python 04-predict.py` # ~4.5 hours
* `gzip pred_ftrl.txt`
* submit `pred_ftrl.txt.gz`s (~1.5 hours)

