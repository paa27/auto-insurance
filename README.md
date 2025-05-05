# Auto Insurance Pricing Model

## Overview
- This project implements a solution for the auto insurance pricing challenge described in `data/assignment.pdf`. 
- The final solution script is provided in `notebooks/pipeline.ipynb`. 
- The actual prediction file can be found in `data/test_evaluation.csv`
- Slides to be used during interview can be found in `slides.pdf`.

## Pricing strategy

~~~
P(Q): pricing model at quantile Q.

q: quantile used for pricing estimate.
q_high: upper quantile for uncertainty estimate.
q_low: lower quantile for uncertainty estimate.
s_lim: uncertainty cutoff. Above s_lim, no price is provided.

Strategy:

    s = P(q_high) - P(q_low)
    s = (s - s.mean())/s.std()

    if s < s_lim
        Offer P(q)

    else:
        No price offered

~~~



