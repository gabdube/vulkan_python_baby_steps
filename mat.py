from ctypes import c_float, Structure
from itertools import chain
from math import tan


class Mat4(Structure):
    
    __slots__ = ('staging', 'data')
    _fields_ = (('data', c_float*16),)

    def __init__(self):
        self.staging = staging = [
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

        obj.staging = staging = [
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

        obj.staging = staging = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [x, y, z, 1.0]
        ]

        obj.data[::] = (c_float*16)(*chain(*staging))

        return obj