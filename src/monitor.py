import pandas as pd
from alibi_detect.cd import MMDDrift
import numpy as np

def detect_drift(X_ref, X_new, p_val=0.05):
    cd = MMDDrift(X_ref, p_val=p_val)
    preds = cd.predict(X_new)
    return preds
