from __future__ import division, print_function
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings

import os
from keras.models import load_model
model = load_model("modelcheck/bestmodel.h5")
