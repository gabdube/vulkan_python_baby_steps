from math import sqrt


class Vec3(object):

    @staticmethod
    def normalize(vector):
        v, dst = vector, [0, 0, 0]
        length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        if (length > 0.00001):
            dst[0] = v[0] / length
            dst[1] = v[1] / length
            dst[2] = v[2] / length

        return dst
