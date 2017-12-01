# Ad Placement Challenge

The winning solution to the [Ad Placement Challenge](https://www.crowdai.org/challenges/nips-17-workshop-criteo-ad-placement-challenge) by Criteo for the 
[NIPS'17 Causal Inference and Machine Learning Workshop](https://sites.google.com/view/causalnips2017)

The solution is simple:

* Take the data as is, train a FTRL model (using [libftrl-python](https://github.com/alexeygrigorev/libftrl-python))
* Post-process the predictions:
  * Apply the sigmoid fuction to the predictions and scale the result by some constant value
  * For each group add +15 to the max value
