from .. import vk
from .utils import check_ctypes_members, array_pointer, array, sequence_to_array
from ctypes import byref, c_uint32


def component_mapping(**kwargs):
    check_ctypes_members(vk.ComponentMapping, (), kwargs.keys())
    return vk.ComponentMapping(
        r = kwargs.get('r', vk.COMPONENT_SWIZZLE_R),
        g = kwargs.get('g', vk.COMPONENT_SWIZZLE_G),
        b = kwargs.get('b', vk.COMPONENT_SWIZZLE_B),
        a = kwargs.get('a', vk.COMPONENT_SWIZZLE_A),
    )


def image_subresource_range(**kwargs):
    check_ctypes_members(vk.ImageSubresourceRange, (), kwargs.keys())
    return vk.ImageSubresourceRange(
        aspect_mask = kwargs.get('aspect_mask', vk.IMAGE_ASPECT_COLOR_BIT),
        base_mip_level = kwargs.get('base_mip_level', 0),
        level_count = kwargs.get('level_count', 1),
        base_array_layer = kwargs.get('base_array_layer', 0),
        layer_count = kwargs.get('layer_count', 1),
    )


def image_view_create_info(**kwargs):
    check_ctypes_members(vk.ImageViewCreateInfo, ('image', 'format'), kwargs.keys())

    return vk.ImageViewCreateInfo(
        type = vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        next = None,
        flags = 0,
        image = kwargs['image'],
        view_type = kwargs.get('view_type', vk.IMAGE_VIEW_TYPE_2D),
        format = kwargs['format'],
        components = kwargs.get('components', component_mapping()),
        subresource_range = kwargs.get('subresource_range', image_subresource_range())
    )


def create_image_view(api, device, info):
    view = vk.ImageView(0)
    result = api.CreateImageView(device, byref(info), None, byref(view))
    if result != vk.SUCCESS:
        raise RuntimeError(f"Failed to create an image view: {result}")

    return view


def destroy_image_view(api, device, view):
    api.DestroyImageView(device, view, None)


def image_create_info(**kwargs):
    check_ctypes_members(vk.ImageCreateInfo, ('format', 'extent', 'usage'), kwargs.keys())

    queue_family_indices, queue_family_indices_ptr, queue_family_index_count = sequence_to_array(kwargs.get('queue_family_indices'), c_uint32)

    return vk.ImageCreateInfo(
        type = vk.STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        next = None,
        flags = 0,
        image_type = kwargs.get('image_type', vk.IMAGE_TYPE_2D),
        format = kwargs['format'],
        extent = kwargs['extent'],
        mip_levels = kwargs.get('mip_levels', 1),
        array_layers = kwargs.get('array_layers', 1),
        samples = kwargs.get('samples', vk.SAMPLE_COUNT_1_BIT),
        tiling = kwargs.get('tiling', vk.IMAGE_TILING_OPTIMAL),
        usage = kwargs['usage'],
        sharing_mode = kwargs.get('sharing_mode', vk.SHARING_MODE_EXCLUSIVE),
        queue_family_index_count = queue_family_index_count,
        queue_family_indices = queue_family_indices_ptr,
        initial_layout = kwargs.get('initial_layout', vk.IMAGE_LAYOUT_UNDEFINED)
    )


def image_memory_requirements(api, device, image):
    req = vk.MemoryRequirements()
    api.GetImageMemoryRequirements(device, image, byref(req))
    return req


def bind_image_memory(api, device, image, memory, offset=0):
    result = api.BindImageMemory(device, image, memory, offset)
    if result != vk.SUCCESS:
        raise RuntimeError("Failed to bind memory to image")


def create_image(api, device, info):
    image = vk.Image(0)
    result = api.CreateImage(device, byref(info), None, byref(image))
    if result != vk.SUCCESS:
        raise RuntimeError("Failed to create an image")

    return image


def destroy_image(api, device, image):
    api.DestroyImage(device, image, None)
