import numpy as np
import json
# x: right, y: up, z: backward


def labnotation_dictionary():
    # polar angle
    theta = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
    theta_label = ['place_high', 'high', 'middle', 'low', 'place_low']
    # azimuthal angle
    phi = np.array([0,
                    np.pi / 4,
                    np.pi / 2,
                    3 * np.pi / 4,
                    np.pi,
                    5 * np.pi / 4,
                    3 * np.pi / 2,
                    7 * np.pi / 4])
    phi_label = ['forward',
                 'rightforward',
                 'right',
                 'rightbackward',
                 'backward',
                 'leftbackward',
                 'left',
                 'leftforward']
    dictionary = {}
    for i, label in enumerate(theta_label):
        for j, label2 in enumerate(phi_label):
            vector = [np.sin(theta[i]) * np.sin(phi[j]),
                      np.cos(theta[i]),
                      -np.sin(theta[i]) * np.cos(phi[j])]
            if label == 'place_high' or label == 'place_low':
                dictionary[label] = vector
            else:
                dictionary[label2 + '_' + label] = vector
    return dictionary


if __name__ == '__main__':
    dictionary = labnotation_dictionary()
    with open('laban_orientations.json', 'w') as f:
        json.dump(dictionary, f, indent=4)
