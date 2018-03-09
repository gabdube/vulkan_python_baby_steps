# -*- coding: utf-8 -*-
"""
Texture class. Load texture object in the application and onto the GPU.
Currently, only a single type of texture can be loaded: DTX3(BC2) encoded textures using the KTX format.
"""

from vulkan import vk, helpers as hvk
from ctypes import Structure, c_ubyte, c_uint32, sizeof, memmove
from collections import namedtuple
import struct

Ktx10HeaderData = hvk.array(c_ubyte, 12, (0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A))
GL_COMPRESSED_RGBA_S3TC_DXT3_EXT = 33778


MipmapData = namedtuple('MipmapData', ('offset', 'size', 'width', 'height'))
GpuTexture = namedtuple('GpuTexture', ('image', 'view', 'sampler', 'layout'))


class KtxHeader(Structure):
    """
    The header of a ktx file
    """
    _fields_ = (
        ('id', c_ubyte*12),
        ('endianness', c_uint32),
        ('gl_type', c_uint32),
        ('gl_type_size', c_uint32),
        ('gl_format', c_uint32),
        ('gl_internal_format', c_uint32),
        ('gl_base_internal_format', c_uint32),
        ('pixel_width', c_uint32),
        ('pixel_height', c_uint32),
        ('pixel_depth', c_uint32),
        ('number_of_array_elements', c_uint32),
        ('number_of_faces', c_uint32),
        ('number_of_mipmap_levels', c_uint32),
        ('bytes_of_key_value_data', c_uint32),
    )

    def __repr__(self):
        return repr({n: v for n, v in [(n[0], getattr(self, n[0])) for n in self._fields_]})


class KtxTexture(object):
    """
    Warning: This class only implements loading functions for the images formats used in this project.
    Any other usage will most likely not work.

    Texture arrays and cubic texture are not supported and the texture endianess must match the system endianess.
    """

    def __init__(self, fname, header, data):
        self.file_name = fname

        # Main texture data
        self.width = header.pixel_width
        self.height = max(header.pixel_height, 1)
        self.depth = max(header.pixel_depth, 1)
        self.mips_level = max(header.number_of_mipmap_levels, 1)
        self.array_element = max(header.number_of_array_elements, 1)
        self.faces = max(header.number_of_faces, 1)
        self.target = KtxTexture.header_target(header)
        self.format = KtxTexture.header_format(header)
        self.texel_size = KtxTexture.header_texel_size()
        self.data = bytearray()

        # Mipmap data
        self.mipmaps = []

        if self.array_element > 1 or self.faces > 1:
            raise NotImplementedError("Texture array and cubic textures are not yet implemented.")

        # Load the texture data
        data_offset = local_offset = 0
        mip_extent_width, mip_extent_height = self.width, self.height
        for i in range(self.mips_level):
            mipmap_size = struct.unpack_from("=I", data, data_offset)[0]
            data_offset += 4

            self.data.extend(data[data_offset:data_offset+mipmap_size])
            self.mipmaps.append(MipmapData(local_offset, mipmap_size, mip_extent_width, mip_extent_height))

            mip_extent_width //= 2
            mip_extent_height //= 2
            data_offset += mipmap_size
            local_offset += mipmap_size

    def data_ptr(self):
        """Return a pointer to the image raw data by concatenating the image and its mipmap data"""
        return hvk.array(c_ubyte, len(self.data), self.data)

    @staticmethod
    def header_target(header):
        """
        Get the target of a ktx texture based on the header data
        Cube & array textures not implemented

        :param header: The header loaded with `load`
        :return:
        """

        if header.pixel_height == 0:
            return vk.IMAGE_TYPE_1D
        elif header.pixel_depth > 0:
            return vk.IMAGE_TYPE_3D

        return vk.IMAGE_TYPE_2D

    @staticmethod
    def header_format(header):
        """
        Check the format of the texture. Only BC2 UNORM (aka DTX3) is supported.

        :param header: The parsed file header
        :return: The vulkan format identifier
        """
        h = header
        is_compressed = h.gl_type == 0 and h.gl_type_size == 1 and h.gl_format == 0
        is_bc2 = h.gl_internal_format == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT

        if is_compressed and is_bc2:
            return vk.FORMAT_BC2_SRGB_BLOCK
        else:
            raise ValueError("The format of the texture \"{}\" is not BC2 UNORM".format())

    @staticmethod
    def header_texel_size():
        # Texel size of GL_COMPRESSED_RGBA_S3TC_DXT3_EXT is 16 bytes
        return 16

    @staticmethod
    def block_size(format_):
        """
        Return the block size of the file format. Currently, only `FORMAT_BC2_SRGB_BLOCK` is defined.
        :param format_: The file format return by `header_format`
        :return: The block size as an int
        """

        block_size_table = {
            vk.FORMAT_BC2_SRGB_BLOCK: 16
        }

        size = block_size_table.get(format_)
        if size is not None:
            return size
        else:
            raise ValueError("Block size of format {} is not defined".format(size))

    @staticmethod
    def load(filename):
        """
        Load and parse a KTX texture in memory

        :param filename: The relative path of the file to load
        :return: A KtxTexture texture object
        """

        # File size check
        data, length = load_file(filename)
        if length < sizeof(KtxHeader):
            msg = "The file {} is valid: length inferior to the ktx header"
            raise IOError(msg.format(filename))

        # Header check
        header = hvk.utils.bytes_to_cstruct(data, KtxHeader)
        if header.id[::] != Ktx10HeaderData[::]:
            msg = "The file {} is not valid: header do not match the ktx header"
            raise IOError(msg.format(filename))

        offset = sizeof(KtxHeader) + header.bytes_of_key_value_data
        texture = KtxTexture(filename, header, data[offset::])

        return texture


def load_file(filename):
    with open(filename, 'rb') as f:
        data = f.read()

    return data, len(data)
