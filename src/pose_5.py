import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))

def add_pose(graph, initial_estimate, pose_5):
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    print(result)
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = "d"
    best_landmark = 1

    g  = gtsam.NonlinearFactorGraph(graph)
    ie = gtsam.Values(initial_estimate)

    pose_5 = pose_options[best_pose]
    g, ie  = add_pose(g, ie, pose_5)
    result = optimize(g, ie)
    g      = add_landmark_measurement(g, result, pose_5, best_landmark)
    result = optimize(g, ie)

    marginals = gtsam.Marginals(g, result)
    sum_of_marginals = (marginals.marginalCovariance(L(1)).sum() +
                        marginals.marginalCovariance(L(2)).sum())
    return best_pose, best_landmark, sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = "b"
    best_landmark = 2

    g  = gtsam.NonlinearFactorGraph(graph)
    ie = gtsam.Values(initial_estimate)

    pose_5 = pose_options[best_pose]
    g, ie  = add_pose(g, ie, pose_5)
    result = optimize(g, ie)
    g      = add_landmark_measurement(g, result, pose_5, best_landmark)
    result = optimize(g, ie)

    true_xy = {1: (0.0, 0.0), 2: (2.0, 0.0), 3: (4.0, 0.0)}
    list_of_errors = []
    for i in [1, 2, 3]:
        p = result.atPose2(X(i))
        tx, ty = true_xy[i]
        list_of_errors.append(abs(p.x() - tx) + abs(p.y() - ty))

    sum_of_errors = sum(list_of_errors)
    return best_pose, best_landmark, sum_of_errors