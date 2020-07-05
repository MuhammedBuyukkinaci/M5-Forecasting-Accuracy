# M5-Forecasting-Accuracy
My notebooks and scripts on M5 Forecasting - Accuracy Competition on Kaggle

# What I learned

1) Different models with respect to state, store and department should be set up. Their predictions can be used in ensembling.

2) Tweedie and Poisson objective can be used in Lightgbm.

3) For time series problems, using only latest years' data is crucial.

4) Lag features and rolling features of lag features can be created.

5) It would be sensible to use multiple validation schemes(2014, 2015 ,2016).

6) Last digit of price can be used as a feature.

7) Adding a flag which means the product is new or not can be considerable.

8) The method named pct_change can be used in pandas series rolling.

9) Assigning weights to training samples in lightgbm can be used.
