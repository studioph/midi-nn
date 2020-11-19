import numpy as np

"""
A basic predictor that simply uses the velocity of the previous note as the prediction, 
or the average velocity of the whole sequence for every value.
"""
class BasicVelocityPredictor:
    def __init__(self):
        pass

    def predict(notes):
        return notes[-1:] + notes[:-1]

    def predict_avg(notes):
        avg_velocity = np.mean(notes)
        return np.full((len(notes)), avg_velocity).tolist()