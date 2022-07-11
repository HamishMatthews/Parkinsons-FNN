import os
import sys
from tensorflow import keras
import pandas as pd
import numpy as np
import parselmouth
from parselmouth.praat import call
import re
from pyrpde import rpde
from scipy.io.wavfile import read
import time
import nolds
import fathon
from fathon import fathonUtils as fu

### basic analysis using existing data ###
removed_cols = ['status', 'name', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
model = keras.models.load_model('model', compile = True)



def get_report(voice_file, f0min, f0max, unit, startTime, endTime):
    sound = parselmouth.Sound(voice_file) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    pulses = parselmouth.praat.call([sound, pitch], "To PointProcess (cc)")
    voice_report_str = parselmouth.praat.call([sound, pitch, pulses], "Voice report", startTime, endTime, 75, 600, 1.3, 1.6, 0.03, 0.45)
    voice_report_str.replace('undefined', '0')
    report_vals = re.findall(r'-?\d+\.?\d*', voice_report_str)
    #print(voice_report_str)
    #report = [s[21], s[22]+'E'+s[23],s[24],s[26],s[27],s[28],s[29],s[31],s[33],s[35]]
    return report_vals


def analyse_rpde(voice_file, test=False):
    rate, data = read(voice_file)
    # https://github.com/bootphon/sustained-phonation-features/blob/master/Sustained_Phonation.ipynb
    if test:
        tests = [0, 10, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 5000, None]
        for i in tests:
            start_time = time.time()
            entropy, histogram = rpde(data, tmax=i, parallel=False)
            print(i, " - ", entropy, " --- %s seconds ---" % (time.time() - start_time))
            return entropy
    # no tests - use 5000
    start_time = time.time()
    entropy, histogram = rpde(data, tmax=500, parallel=False)
    print("RPDE Analysis Time - tmax: ", 500, " --- %s seconds ---" % (time.time() - start_time))
    return entropy

# currently resized to 5000
def analyse_d2(voice_file):
    rate, data = read(voice_file)
    print("Analysing D2")
    sample = np.resize(data, 5000)
    start_time = time.time()
    d2 = nolds.corr_dim(sample, emb_dim=10)
    print("D2 analysis time --- %s seconds --- " % (time.time() - start_time), d2)
    return d2


# This is the function to measure voice pitch
def analyse_voice(voice_file, f0min, f0max, unit):

    #requirements
    sound = parselmouth.Sound(voice_file)  # read the sound
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max)  # create a praat pitch object

    #variables
    meanF0 = call(pitch, "Get mean", 0, 0, unit)  # get mean pitch
    minF0 = call(pitch, "Get minimum", 0, 0, unit, "Parabolic")  # get min pitch
    maxF0 = call(pitch, "Get maximum", 0, 0, unit, "Parabolic")  # get max pitch
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer = call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer = call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    nhr = get_report(voice_file, 75, 500, "Hertz", 0, 0)[-2]
    hnr = call(harmonicity, "Get mean", 0, 0)

    #excluding as computation time is too great or unknown methodologies
    #RPDE = analyse_rpde(voice_file)
    #D2 = analyse_d2(voice_file)
    #DFA = analyse_DFA(voice_file)
    #PPE - needs development

    return {"meanF0": meanF0, "minF0": minF0, "maxF0": maxF0, "localJitter": localJitter,
            "localabsoluteJitter": localabsoluteJitter, "rapJitter": rapJitter, "ppq5Jitter": ppq5Jitter,
            "ddpJitter": ddpJitter, "localShimmer": localShimmer, "localdbShimmer": localdbShimmer,
            "apq3Shimmer": apq3Shimmer, "aqpq5Shimmer": aqpq5Shimmer, "apq11Shimmer": apq11Shimmer,
            "ddaShimmer": ddaShimmer, "nhr": nhr, "hnr": hnr}

def analyse_DFA(voice_file, plot=False):
    rate, data = read(voice_file)
    a = fu.toAggregated(data)
    pydfa = fathon.DFA(a)

    winSizes = fu.linRangeByStep(10, 2000)
    revSeg = True
    polOrd = 3

    n, F = pydfa.computeFlucVec(winSizes, revSeg=revSeg, polOrd=polOrd)
    H, H_intercept = pydfa.fitFlucVec()

    return H


def main():
    if(len(sys.argv) > 2):
        print("Too many arguments.")
        return None

    if len(sys.argv) == 1:
        print("no file given - predicting using input data.")
        park = pd.read_csv("data/parkinsons.data")
        a = park.loc[:, ~park.columns.isin(removed_cols)].to_numpy()[33]
        test_pred = model.predict(np.expand_dims(a, axis=0))
        print(test_pred)
        return test_pred
    #analyse real file
    fn = sys.argv[1]
    if os.path.exists(fn):
        results = analyse_voice(fn, 75, 1000, "Hertz")
        results_arr = np.expand_dims(np.array(list(results.values()), dtype='float32'), axis=0)
        real_test = model.predict(results_arr)
        print("results:", real_test)
        return real_test


if __name__== "__main__" :
    main()
