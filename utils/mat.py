from ctypes import c_float, Structure
from itertools import chain
from math import tan, sin, cos, sqrt
from sys import float_info

buffer_type = c_float*16

class Mat4(Structure):
    
    __slots__ = ('staging', 'data')
    _fields_ = (('data', buffer_type),)

    def __init__(self):
        staging = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )

        self.data[::] = buffer_type(*chain(*staging))

    @classmethod
    def perspective(cls, fovy, aspect, near, far):
        obj = super(Mat4, cls).__new__(cls)

        f = 1.0 / tan(fovy / 2)
        nf = 1.0 / (near - far)

        staging = (
            (f / aspect, 0.0, 0.0, 0.0),
            (0.0, f, 0.0, 0.0),
            (0.0, 0.0, (far + near) * nf, -1.0),
            (0.0, 0.0, (far * near * 2) * nf, 0.0)
        )

        obj.data[::] = buffer_type(*chain(*staging))

        return obj

    @classmethod
    def from_translation(cls, x, y, z):
        obj = super(Mat4, cls).__new__(cls)

        staging = (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (x, y, z, 1.0)
        )

        obj.data[::] = buffer_type(*chain(*staging))

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

        obj.data[::] = buffer_type(*chain(r1, r2, r3, r4))

        return obj

    def rotate(self, rad, axis):
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

        # Load the matrix into local values because it's faster than operating directly into the ctypes buffer
        data = self.data
        a00, a01, a02, a03 = data[0:4]
        a10, a11, a12, a13 = data[4:8]
        a20, a21, a22, a23 = data[8:12]

        #Construct the elements of the rotation matrix
        b00, b01, b02 = (x * x * t + c), (y * x * t + z * s), (z * x * t - y * s)
        b10, b11, b12 = ( x * y * t - z * s), (y * y * t + c), (z * y * t + x * s)
        b20, b21, b22 = (x * z * t + y * s), (y * z * t - x * s), (z * z * t + c)

        # Perform rotation-specific matrix multiplication

        staging = (
            (
                a00 * b00 + a10 * b01 + a20 * b02,
                a01 * b00 + a11 * b01 + a21 * b02,
                a02 * b00 + a12 * b01 + a22 * b02,
                a03 * b00 + a13 * b01 + a23 * b02
            ),
            (
                a00 * b10 + a10 * b11 + a20 * b12,
                a01 * b10 + a11 * b11 + a21 * b12,
                a02 * b10 + a12 * b11 + a22 * b12,
                a03 * b10 + a13 * b11 + a23 * b12
            ),
            (
                a00 * b20 + a10 * b21 + a20 * b22,
                a01 * b20 + a11 * b21 + a21 * b22,
                a02 * b20 + a12 * b21 + a22 * b22,
                a03 * b20 + a13 * b21 + a23 * b22
            ),
            data[12:16]
        )

        self.data[::] = buffer_type(*chain(*staging))
        
        return self
