
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Robot rotates 45 degrees, moves 2 m, then rotates 45 degrees more.
    # In X(3)'s local frame: dx = sqrt(2), dy = sqrt(2), dtheta = pi/2
    relative_pose = gtsam.Pose2(math.sqrt(2), math.sqrt(2), math.pi / 2)
 
    # Optimize the existing graph first so X(3) is accurate before composing.
    result = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate).optimize()
 
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), relative_pose, ODOMETRY_NOISE))
    initial_estimate.insert(X(4), result.atPose2(X(3)).compose(relative_pose))
    
    return graph, initial_estimate