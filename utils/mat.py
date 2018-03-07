from ctypes import c_float, Structure
from itertools import chain
from math import tan, sin, cos, sqrt
from sys import float_info


class Mat4(Structure):
    
    __slots__ = ('staging', 'data')
    _fields_ = (('data', c_float*16),)

    def __init__(self):
        staging = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]

        self.data[::] = (c_float*16)(*chain(*staging))

    @classmethod
    def perspective(cls, fovy, aspect, near, far):
        obj = super(Mat4, cls).__new__(cls)

        f = 1.0 / tan(fovy / 2)
        nf = 1.0 / (near - far)

        staging = [
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) * nf, -1.0],
            [0.0, 0.0, (far * near * 2) * nf, 0.0]
        ]

        obj.data[::] = (c_float*16)(*chain(*staging))

        return obj

    @classmethod
    def from_translation(cls, x, y, z):
        obj = super(Mat4, cls).__new__(cls)

        staging = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0]
        ]

        obj.data[::] = (c_float*16)(*chain(*staging))

        return obj

    @classmethod
    def from_rotation(cls, rad, axis):
        obj = super(Mat4, cls).__new__(cls)

        x, y, z = axis
        length = sqrt(x * x + y * y + z * z)

        if abs(length) < float_info.epsilon:
            raise ValueError()

        length = 1 / length
        x *= length
        y *= length
        z *= length

        s = sin(rad)
        c = cos(rad)
        t = 1 - c

        r1, r2, r3, r4 = ([0.0]*4 for _ in range(4))

        r1[0] = x * x * t + c
        r1[1] = y * x * t + z * s
        r1[2] = z * x * t - y * s

        r2[0] = x * y * t - z * s
        r2[1] = y * y * t + c
        r2[2] = z * y * t + x * s

        r3[0] = x * z * t + y * s
        r3[1] = y * z * t - x * s
        r3[2] = z * z * t + c

        r4[3] = 1.0

        staging = [r1, r2, r3, r4]
        obj.data[::] = (c_float*16)(*chain(*staging))

        return obj