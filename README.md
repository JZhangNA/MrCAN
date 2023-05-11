# MrCAN

This code is the implementation of the paper "MrCAN: Multi-relations aware convolutional attention network for multivariate time series forecasting".

## Requirements

The recommended requirements for MrCAN are specified as follows:
* Python 3.8
* torch==1.13.1
* numpy==1.22.4
* pandas==1.5.3
* scikit-learn==1.2.1

The dependencies can be installed by:
```bash
pip install -r requirements.txt
```

## Data

The datasets can be obtained and put into `dataset/` folder

* [SML2010](https://archive.ics.uci.edu/ml/datasets/sml2010)
* [Highways Traffic](http://data.gov.uk/dataset/highways-england-network-journey-time-and-traffic-flow-data)
* [PM2.5](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
* [EEG](https://archive.ics.uci.edu/ml/datasets/EEG+Steady-State+Visual+Evoked+Potential+Signals)
* [ETT](https://github.com/zhouhaoyi/ETDataset).

## Usage

To train and evaluate MrCAN on a dataset, run the following command:

```train & evaluate
python main.py
```
