try:
    import biorbd
except ModuleNotFoundError:
    pass
# try:
#     from casadi import MX, Function, horzcat
# except ModuleNotFoundError:
#     pass
import numpy as np


def kalman_func(markers, model, return_q_dot=True, kalman=None):
    markers_over_frames = []
    if not kalman:
        freq = 100  # Hz
        params = biorbd.KalmanParam(freq)
        kalman = biorbd.KalmanReconsMarkers(model, params)

    q = biorbd.GeneralizedCoordinates(model)
    q_dot = biorbd.GeneralizedVelocity(model)
    qd_dot = biorbd.GeneralizedAcceleration(model)
    for i in range(markers.shape[2]):
        markers_over_frames.append([biorbd.NodeSegment(m) for m in markers[:, :, i].T])

    q_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
    q_dot_recons = np.ndarray((model.nbQ(), len(markers_over_frames)))
    for i, targetMarkers in enumerate(markers_over_frames):
        kalman.reconstructFrame(model, targetMarkers, q, q_dot, qd_dot)
        q_recons[:, i] = q.to_array()
        q_dot_recons[:, i] = q_dot.to_array()

    # compute markers from
    if return_q_dot:
        return q_recons, q_dot_recons
    else:
        return q_recons


# def markers_fun(biorbd_model, q=None, eigen_backend=False):
#     if eigen_backend:
#         return [biorbd_model.markers(q)[i].to_array() for i in range(biorbd_model.nbMarkers())]
#     else:
#         qMX = MX.sym("qMX", biorbd_model.nbQ())
#         return Function(
#             "markers",
#             [qMX],
#             [horzcat(*[biorbd_model.markers(qMX)[i].to_mx() for i in range(biorbd_model.nbMarkers())])],
#         )