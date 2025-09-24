import warp as wp
import numpy as np
from DataTypes import *

class Muscle():
    '''
        Muscle class on Host.
    '''
            
    def __init__(self, bodies, via_points, index):  
        self.bodies = bodies
        self.via_points = via_points
        self.index = index
 
class MuscleSpring(Muscle):
    def __init__(self, bodies, points, stiffness, rest_length, index):
        if len(points) < 4:
            raise ValueError("At least 2 via points are required to create a MuscleSpring. Points list should look like [Origin Via_point1 Via_point2 ... Insertion].")
        super().__init__(bodies, points[1:-1], index)
        rest_length -= wp.norm_l2(points[0] - points[1]) + wp.norm_l2(points[-2] - points[-1])
        self.stiffness = stiffness
        self.rest_length = rest_length

class Museculotendon(Muscle):
    def __init__(self, bodies, points, muscle_params, index):
        if len(points) < 4:
            raise ValueError("At least 2 via points are required to create a MuscleSpring. Points list should look like [Origin Via_point1 Via_point2 ... Insertion].")
        super().__init__(bodies, points[1:-1], index)
        self.muscle_params = muscle_params

class MuscleParams:
    def __init__(self):
        self.lMopt = 0.1      # optimal muscle length
        self.lTslack = 0.2    # tendon slack length
        self.vMmax = 10.0     # maximum contraction velocity
        self.alphaMopt = 0.0  # pennation angle at optimal muscle length
        self.fMopt = 1000    # peak isometric force
        self.beta = 0.1       # damping
        self.h = self.lMopt * np.sin(self.alphaMopt)
        self.compliance = 1.0 / self.fMopt  # XPBD compliance