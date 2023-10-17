#Metric Correlation

The files in this folder should be used to compute correlation via bootstrapping between human evaluation and 
automatic metrics.

Each file can be run as follows:

    filename.py parameters_year.json

In metrics results we have already the precomputed metrics.

In datasets we have WebNLG2017 and WebNLG2020 annotations.

To compute the correlations run:

    correlations.py parameters_year.json