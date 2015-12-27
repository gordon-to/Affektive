import sys
import requests
import json
from flask import Flask, request, jsonify, session, g, redirect, url_for, abort, render_template, flash
from contextlib import closing
import validation

# Import Pickle
import pickle


import numpy as np
# ML Imports
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


data = pd.read_csv('output.csv')
print data.gsr
print data[data['gsr'] > 1000]
