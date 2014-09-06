import bisect
import numpy as np

import matplotlib.pyplot as plt

from lie import SO3


def main():
    orientation_data = np.loadtxt('out/frame_orientations.txt')
    accel_data = np.loadtxt('out/accelerometer.txt')

    frame_timestamps = orientation_data[:,0]
    frame_orientations = orientation_data[:, 1:].reshape((-1, 3, 3))

    accel_timestamps = accel_data[:,0]
    accel_readings = accel_data[:,1:]

    accel_orientations = []
    for accel_time in accel_timestamps:
        frame_index = bisect.bisect_left(frame_timestamps, accel_time)
        assert 0 < frame_index < len(frame_timestamps), 't='+accel_time
        t0 = frame_timestamps[frame_index-1]
        r0 = frame_orientations[frame_index-1]
        t1 = frame_timestamps[frame_index]
        r1 = frame_orientations[frame_index]
        w01 = SO3.log(np.dot(r1, r0.T))
        a = (accel_time - t0) / (t1 - t0)
        assert 0 <= a <= 1
        accel_orientations.append(np.dot(SO3.exp(a*w01), r0))

    frame_ws = []
    accel_ws = []
    rbase = frame_orientations[0]
    for r in frame_orientations:
        frame_ws.append(SO3.log(np.dot(r, rbase.T)))
    for r in accel_orientations:
        accel_ws.append(SO3.log(np.dot(r, rbase.T)))

    np.savetxt('out/accel_orientations.txt',
               np.hstack((accel_timestamps[:,None], np.reshape(accel_orientations, (-1, 9)))))

    plt.clf()
    plt.hold('on')
    plt.plot(frame_timestamps, np.asarray(frame_orientations).reshape((-1, 9)))
    plt.plot(accel_timestamps, np.asarray(accel_orientations).reshape((-1, 9)), '.')
    #plt.plot(frame_timestamps, map(np.linalg.norm, frame_ws))
    #plt.plot(accel_timestamps, map(np.linalg.norm, accel_ws), '.')
    plt.show()


if __name__ == '__main__':
    main()
