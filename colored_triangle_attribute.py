"""
Render a static colored triangle using shader attributes.

Vertex shader use a "position" attribute of type `vec3` and a "color" attribute of type `vec3`.
Vertex shader attributes are tightly packed and use different bindings
No modification (view, model, frustrum) are applied to the vertex.

No uniforms/buffers defined in the shaders. 

Fragment color is set to the attribute value
"""


from vulkan import vk, helpers as hvk
from vulkan.debugger import Debugger
from system.window import Window

import platform, time
from enum import IntFlag
from collections import namedtuple
from ctypes import c_ushort, c_float, sizeof, memmove, byref

# Global data
window = None

api = None
instance = None
debugger = None
surface = None

physical_device = None
render_queue_family = None
device = None
render_queue = None

swapchain_image_format = None
swapchain = None
swapchain_images = None
swapchain_image_views = None

depth_format = None
depth_stencil = None
depth_alloc = None
depth_view = None

staging_pool = None
staging_cmd = None
staging_fence = None

mesh_positions = {'data': None, 'size': None, 'offset': None}
mesh_colors = {'data': None, 'size': None, 'offset': None}
mesh_indices = {'data': None, 'size': None, 'offset': None}
total_mesh_size = None

staging_mesh_buffer = None
staging_mesh_memory = None

mesh_buffer = None
mesh_memory = None

render_pass = None
framebuffers = None
shader_modules = None
stage_infos = None
descriptor_set_layout = None
pipeline_layout = None
vertex_bindings = None
vertex_attributes = None
pipeline = None
pipeline_cache = None

drawing_pool = None
cmd_draw = None
image_ready = None
rendering_done = None
render_fences = None

# Typing setup
Queue = namedtuple("Queue", ("handle", "family"))


# Helpers functions
def find_memory_type(heap_flags, type_flags):
    F, type_flags = IntFlag, hvk.MemoryPropertyFlag(type_flags)

    props = hvk.physical_device_memory_properties(api, physical_device)
    types = props.memory_types[:props.memory_type_count]
    heaps = props.memory_heaps[:props.memory_heap_count]

    # Find the memory heap that satisfy heap_flags
    heap_indices = tuple(i for i, h in enumerate(heaps) if F(heap_flags) in F(h.flags))
    if heap_indices is None:
        return

    # Find a memory type in the selected heaps that satisfy type_flags
    type_index = None
    for i, memory_type in enumerate(types):
        heap_matches = memory_type.heap_index in heap_indices
        type_flags_matches = type_flags in hvk.MemoryPropertyFlag(memory_type.property_flags)
        if heap_matches and type_flags_matches:
            type_index = i
            break

    return type_index


# Window setup
def create_window():
    global window
    window = Window(width=800, height=600)

#
# BASIC SETUP (instance / device / swapchain)
#

def create_instance():
    global api, instance

    # Layers
    # Check if debugging layers are present
    layer_names = [l[0] for l in hvk.enumerate_layers()]
    if "VK_LAYER_LUNARG_core_validation" in layer_names and "VK_LAYER_LUNARG_standard_validation" in layer_names:
        layers = ("VK_LAYER_LUNARG_core_validation", "VK_LAYER_LUNARG_standard_validation")
    else:
        layers = ()

    # Extensions
    extensions = ["VK_KHR_surface", "VK_EXT_debug_report"]

    system_name = platform.system()
    if system_name == 'Windows':
        extensions.append('VK_KHR_win32_surface')
    elif system_name == 'Linux':
        extensions.append('VK_KHR_xcb_surface')

    # Api & Instance creation
    api, instance = hvk.create_instance(extensions, layers)


def create_debugger():
    global debugger

    # Debug report setup
    debugger = Debugger(api, instance)
    debugger.start()


def create_surface():
    global surface
    surface = hvk.create_surface(api, instance, window)


def create_device():
    global physical_device, render_queue_family, device

    # Device selection (use the first available)
    physical_devices = hvk.list_physical_devices(api, instance)
    physical_device = physical_devices[0]

    # Queues setup (A single graphic queue)
    queue_families = hvk.list_queue_families(api, physical_device)
    render_queue_family = next(qf for qf in queue_families if IntFlag(vk.QUEUE_GRAPHICS_BIT) in IntFlag(qf.properties.queue_flags))
    render_queue_create_info = hvk.queue_create_info(
        queue_family_index = render_queue_family.index,
        queue_count = 1
    )

    # Device creation
    extensions = ("VK_KHR_swapchain",)
    device = hvk.create_device(api, physical_device, extensions, (render_queue_create_info,))


def get_queues():
    global render_queue

    # Get queue handles
    render_queue_handle = hvk.get_queue(api, device, render_queue_family.index, 0)
    render_queue = Queue(render_queue_handle, render_queue_family)


def create_swapchain():
    global swapchain_image_format, swapchain

    # Swapchain Setup
    caps = hvk.physical_device_surface_capabilities(api, physical_device, surface)
    formats = hvk.physical_device_surface_formats(api, physical_device, surface)
    present_modes = hvk.physical_device_surface_present_modes(api, physical_device, surface)

    if not hvk.get_physical_device_surface_support(api, physical_device, surface, render_queue.family.index):
        raise RuntimeError("Main rendering queue cannot present images to the surface")

    # Swapchain Format
    format_values = tuple(vkf.format for vkf in formats)
    required_formats = [vk.FORMAT_B8G8R8A8_SRGB, vk.FORMAT_B8G8R8A8_UNORM]
    for i, required_format in enumerate(required_formats):
        if required_format in format_values:
            required_formats[i] = format_values.index(required_format)
        else:
            required_formats[i] = None

    selected_format = next((formats[i] for i in required_formats if i is not None), None)
    if selected_format is None:
        raise RuntimeError("Required swapchain image format not supported")

    # Swapchain Extent
    extent = caps.current_extent
    if extent.width == -1 or extent.height == -1:
        width, height = window.dimensions()
        extent.width = width
        extent.height = height

    # Min image count
    min_image_count = 2
    if caps.max_image_count != 0 and caps.max_image_count < min_image_count:
        raise RuntimeError("Minimum image count not met")
    elif caps.min_image_count > min_image_count:
        min_image_count = caps.min_image_count

    # Present mode
    present_mode = vk.PRESENT_MODE_FIFO_KHR
    if vk.PRESENT_MODE_MAILBOX_KHR in present_modes:
        present_mode = vk.PRESENT_MODE_MAILBOX_KHR
    elif vk.PRESENT_MODE_IMMEDIATE_KHR in present_modes:
        present_mode = vk.PRESENT_MODE_IMMEDIATE_KHR

    # Default image transformation
    transform = caps.current_transform
    if IntFlag(vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR) in IntFlag(caps.supported_transforms):
        transform = vk.SURFACE_TRANSFORM_IDENTITY_BIT_KHR

    # Swapchain creation
    swapchain_image_format = selected_format.format
    swapchain = hvk.create_swapchain(api, device, hvk.swapchain_create_info(
        surface = surface,
        image_format = swapchain_image_format,
        image_color_space = selected_format.color_space,
        image_extent = extent,
        min_image_count = min_image_count,
        present_mode = present_mode,
        pre_transform = transform,
        old_swapchain = 0
    ))


def setup_swapchain_image_views():
    global swapchain_images, swapchain_image_views

    # Fetch swapchain images
    swapchain_images = hvk.swapchain_images(api, device, swapchain)

    # Create the swapchain images view
    swapchain_image_views = []
    for image in swapchain_images:
        view = hvk.create_image_view(api, device, hvk.image_view_create_info(
            image = image,
            format = swapchain_image_format
        ))
        swapchain_image_views.append(view)


def setup_swapchain_depth_stencil():
    global depth_format, depth_stencil, depth_alloc, depth_view

    width, height = window.dimensions()

    # Depth stencil attachment setup
    depth_format = None
    depth_formats = (vk.FORMAT_D32_SFLOAT_S8_UINT, vk.FORMAT_D24_UNORM_S8_UINT, vk.FORMAT_D16_UNORM_S8_UINT)
    for fmt in depth_formats:
        prop = hvk.physical_device_format_properties(api, physical_device, fmt)
        if IntFlag(vk.FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) in IntFlag(prop.optimal_tiling_features):
            depth_format = fmt
            break

    if depth_format is None:
        raise RuntimeError("Failed to find a suitable depth stencil format.")

    # Depth stencil image creation
    depth_stencil = hvk.create_image(api, device, hvk.image_create_info(
        format = depth_format,
        extent = vk.Extent3D(width=width, height=height, depth=1),
        usage = vk.IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | vk.IMAGE_USAGE_TRANSFER_SRC_BIT
    ))

    # Depth stencil image memory
    image_memreq = hvk.image_memory_requirements(api, device, depth_stencil)
    mt_index = find_memory_type(vk.MEMORY_HEAP_DEVICE_LOCAL_BIT, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    depth_stencil_size = image_memreq.size * 2
    depth_alloc = hvk.allocate_memory(api, device, hvk.memory_allocate_info(
        allocation_size = depth_stencil_size,
        memory_type_index = mt_index
    ))

    hvk.bind_image_memory(api, device, depth_stencil, depth_alloc)

    # Depth stencil image view
    depth_view = hvk.create_image_view(api, device, hvk.image_view_create_info(
        image = depth_stencil,
        format = depth_format,
        subresource_range = hvk.image_subresource_range(aspect_mask=vk.IMAGE_ASPECT_DEPTH_BIT|vk.IMAGE_ASPECT_STENCIL_BIT)
    ))


def create_staging_command():
    global staging_pool, staging_cmd, staging_fence

    # Create staging resources for uploading
    staging_pool = hvk.create_command_pool(api, device, hvk.command_pool_create_info(
        queue_family_index = render_queue.family.index,
        flags = vk.COMMAND_POOL_CREATE_TRANSIENT_BIT | vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    ))

    staging_cmd = hvk.allocate_command_buffers(api, device, hvk.command_buffer_allocate_info(
        command_pool = staging_pool,
        command_buffer_count = 1
    ))[0]

    staging_fence = hvk.create_fence(api, device, hvk.fence_create_info())


#
# Resources setup
#

def load_mesh():
    global mesh_positions, mesh_indices, mesh_colors, total_mesh_size

    # Store mesh data to into ctypes buffers
    indices = (0, 1, 2)
    indices_count = len(indices)

    positions = ( 
        1.0,  1.0,  0.0,
        -1.0,  1.0,  0.0,
        0.0, -1.0,  0.0
    )

    colors = (
        1.0,  0.0,  0.0,
        0.0,  1.0,  0.0,
        0.0,  0.0,  1.0
    )

    positions_data = (c_float*len(positions))(*positions)
    positions_data_size = sizeof(positions_data)
    positions_data_offset = 0

    colors_data = (c_float*len(colors))(*colors)
    colors_data_size = sizeof(colors_data)
    colors_data_offset = positions_data_size

    indices_data = (c_ushort*len(indices))(*indices)
    indices_data_size = sizeof(indices_data)
    indices_data_offset = colors_data_offset + colors_data_size

    mesh_positions = {
        'data': positions_data,
        'size': positions_data_size,
        'offset': positions_data_offset
    }

    mesh_colors = {
        'data': colors_data,
        'size': colors_data_size,
        'offset': colors_data_offset
    }

    mesh_indices = {
        'data': indices_data,
        'size': indices_data_size,
        'offset': indices_data_offset,
        'count': indices_count
    }

    total_mesh_size = mesh_positions['size'] + mesh_colors['size'] + mesh_indices['size']


def mesh_to_staging():
    global staging_mesh_buffer, staging_mesh_memory

    # Create staging resources
    staging_mesh_buffer = hvk.create_buffer(api, device, hvk.buffer_create_info(
        size = total_mesh_size,
        usage = vk.BUFFER_USAGE_TRANSFER_SRC_BIT
    ))

    staging_req = hvk.buffer_memory_requirements(api, device, staging_mesh_buffer)
    mt_index = find_memory_type(0, vk.MEMORY_PROPERTY_HOST_COHERENT_BIT | vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT)

    staging_mesh_memory = hvk.allocate_memory(api, device, hvk.memory_allocate_info(
        allocation_size = staging_req.size,
        memory_type_index = mt_index
    ))

    hvk.bind_buffer_memory(api, device, staging_mesh_buffer, staging_mesh_memory, 0)

    # Upload mesh to staging data
    data_ptr = hvk.map_memory(api, device, staging_mesh_memory, 0, staging_req.size).value

    memmove(data_ptr + mesh_positions['offset'], byref(mesh_positions['data']), mesh_positions['size'])
    memmove(data_ptr + mesh_colors['offset'], byref(mesh_colors['data']), mesh_colors['size'])
    memmove(data_ptr + mesh_indices['offset'], byref(mesh_indices['data']), mesh_indices['size'])

    hvk.unmap_memory(api, device, staging_mesh_memory)


def mesh_to_device():
    global mesh_buffer, mesh_memory

    # Create mesh resources
    mesh_buffer = hvk.create_buffer(api, device, hvk.buffer_create_info(
        size = total_mesh_size,
        usage = vk.BUFFER_USAGE_TRANSFER_DST_BIT | vk.BUFFER_USAGE_INDEX_BUFFER_BIT | vk.BUFFER_USAGE_VERTEX_BUFFER_BIT
    ))

    mesh_req = hvk.buffer_memory_requirements(api, device, mesh_buffer)
    mt_index = find_memory_type(vk.MEMORY_HEAP_DEVICE_LOCAL_BIT, vk.MEMORY_PROPERTY_DEVICE_LOCAL_BIT)

    mesh_memory = hvk.allocate_memory(api, device, hvk.memory_allocate_info(
        allocation_size = mesh_req.size,
        memory_type_index = mt_index
    ))  

    hvk.bind_buffer_memory(api, device, mesh_buffer, mesh_memory, 0)

    # Upload mesh to device memory (recording)
    hvk.begin_command_buffer(api, staging_cmd, hvk.command_buffer_begin_info())

    region = vk.BufferCopy(src_offset = 0, dst_offset = 0, size = mesh_req.size)
    hvk.copy_buffer(
        api, staging_cmd,
        staging_mesh_buffer,
        mesh_buffer,
        (region,)
    )

    hvk.end_command_buffer(api, staging_cmd)

    # Upload mesh to device memory (submiting) 
    submit_info = hvk.submit_info(command_buffers = (staging_cmd,))
    hvk.queue_submit(api, render_queue.handle, (submit_info,), fence = staging_fence)
    hvk.wait_for_fences(api, device, (staging_fence,))

#
# RENDER SETUP
#

def create_render_pass():
    global render_pass

    # Renderpass attachments setup
    color = hvk.attachment_description(
        format = swapchain_image_format,
        initial_layout = vk.IMAGE_LAYOUT_UNDEFINED,
        final_layout = vk.IMAGE_LAYOUT_PRESENT_SRC_KHR
    )

    depth = hvk.attachment_description(
        format = depth_format,
        initial_layout = vk.IMAGE_LAYOUT_UNDEFINED,
        final_layout = vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    )

    window_attachments = (color, depth)
    color_ref = vk.AttachmentReference(attachment=0, layout=vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
    depth_ref = vk.AttachmentReference(attachment=1, layout=vk.IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)

    # Render pass subpasses
    subpass_info = hvk.subpass_description(
        pipeline_bind_point = vk.PIPELINE_BIND_POINT_GRAPHICS,
        color_attachments = (color_ref,),
        depth_stencil_attachment = depth_ref
    )

    # Renderpass dependencies
    prepare_drawing = hvk.subpass_dependency(
        src_subpass = vk.SUBPASS_EXTERNAL,
        dst_subpass = 0,
        src_stage_mask = vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        dst_stage_mask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        src_access_mask = vk.ACCESS_MEMORY_READ_BIT,
        dst_access_mask = vk.ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    )

    prepare_present = hvk.subpass_dependency(
        src_subpass = 0,
        dst_subpass = vk.SUBPASS_EXTERNAL,
        src_stage_mask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        dst_stage_mask = vk.PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        src_access_mask = vk.ACCESS_COLOR_ATTACHMENT_READ_BIT | vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        dst_access_mask = vk.ACCESS_MEMORY_READ_BIT,
    )

    # Render pass creation
    render_pass = hvk.create_render_pass(api, device, hvk.render_pass_create_info(
        attachments = (color, depth),
        subpasses = (subpass_info,),
        dependencies = (prepare_drawing, prepare_present)
    ))


def create_framebuffers():
    global framebuffers

    width, height = window.dimensions()

    framebuffers = []
    for img in swapchain_image_views:
        framebuffer = hvk.create_framebuffer(api, device, hvk.framebuffer_create_info(
            render_pass = render_pass,
            width = width,
            height = height,
            attachments = (img, depth_view)
        ))

        framebuffers.append(framebuffer)


def create_shaders():
    global shader_modules, stage_infos

    shader_modules, stage_infos = [], []
    shader_sources = {
        vk.SHADER_STAGE_VERTEX_BIT: 'resources/shaders/colored_triangle_attribute/colored_triangle_attribute.vert.spv',
        vk.SHADER_STAGE_FRAGMENT_BIT: 'resources/shaders/colored_triangle_attribute/colored_triangle_attribute.frag.spv'
    }
    for stage, src in shader_sources.items():
        with open(src, 'rb') as f:
            stage_module = hvk.create_shader_module(api, device, hvk.shader_module_create_info(
                code=f.read()
            ))
            shader_modules.append(stage_module)

            stage_infos.append(hvk.pipeline_shader_stage_create_info(
                stage = stage,
                module = stage_module,
            ))


def create_descriptor_set_layout():
    global descriptor_set_layout

    # Setup shader descriptor set 
    info = hvk.descriptor_set_layout_create_info(bindings = ())
    descriptor_set_layout = hvk.create_descriptor_set_layout(api, device, info)


def create_pipeline_layout():
    global pipeline_layout

    # Pipeline layout setup
    pipeline_layout = hvk.create_pipeline_layout(api, device, hvk.pipeline_layout_create_info(
        set_layouts=(descriptor_set_layout,)
    ))


def setup_vertex_input_binding():
    global vertex_bindings, vertex_attributes

    # Setup shader vertex input state
    position_binding = hvk.vertex_input_binding_description(
        binding = 0,
        stride = hvk.utils.format_size(vk.FORMAT_R32G32B32_SFLOAT)
    )

    color_binding = hvk.vertex_input_binding_description(
        binding = 1,
        stride = hvk.utils.format_size(vk.FORMAT_R32G32B32_SFLOAT)
    )

    position_attribute = hvk.vertex_input_attribute_description(
        location = 0,
        binding = 0,
        format = vk.FORMAT_R32G32B32_SFLOAT,
        offset = 0
    )

    color_attribute = hvk.vertex_input_attribute_description(
        location = 1,
        binding = 1,
        format = vk.FORMAT_R32G32B32_SFLOAT,
        offset = 0
    )

    vertex_bindings = (position_binding, color_binding)
    vertex_attributes = (position_attribute, color_attribute)


def create_pipeline():
    global pipeline, pipeline_cache

    width, height = window.dimensions()

    viewport = hvk.viewport(width=width, height=height)
    render_area = hvk.rect_2d(0, 0, width, height)
    pipeline_info = hvk.graphics_pipeline_create_info(
        stages = stage_infos,
        vertex_input_state = hvk.pipeline_vertex_input_state_create_info(
            vertex_binding_descriptions = vertex_bindings,
            vertex_attribute_descriptions = vertex_attributes
        ),
        input_assembly_state = hvk.pipeline_input_assembly_state_create_info(),
        viewport_state = hvk.pipeline_viewport_state_create_info(
            viewports=(viewport,),
            scissors=(render_area,)
        ),
        rasterization_state = hvk.pipeline_rasterization_state_create_info(),
        multisample_state = hvk.pipeline_multisample_state_create_info(),
        depth_stencil_state = hvk.pipeline_depth_stencil_state_create_info(
            depth_test_enable = vk.TRUE,
            depth_write_enable  = vk.TRUE,
            depth_compare_op = vk.COMPARE_OP_LESS_OR_EQUAL,
        ),
        color_blend_state = hvk.pipeline_color_blend_state_create_info(
            attachments = (hvk.pipeline_color_blend_attachment_state(),)
        ),
        layout = pipeline_layout,
        render_pass = render_pass
    )

    pipeline_cache = hvk.create_pipeline_cache(api, device, hvk.pipeline_cache_create_info())
    pipeline = hvk.create_graphics_pipelines(api, device, (pipeline_info,), pipeline_cache)[0]


def create_render_resources():
    global drawing_pool, cmd_draw, image_ready, rendering_done, render_fences

    # Render commands setup
    drawing_pool = hvk.create_command_pool(api, device, hvk.command_pool_create_info(
        queue_family_index = render_queue.family.index,
        flags = vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT
    ))

    cmd_draw = hvk.allocate_command_buffers(api, device, hvk.command_buffer_allocate_info(
        command_pool = drawing_pool,
        command_buffer_count = len(swapchain_images)
    ))

    # Render commands synchronisation resources
    info = hvk.semaphore_create_info()
    image_ready = hvk.create_semaphore(api, device, info)
    rendering_done = hvk.create_semaphore(api, device, info)

    info = hvk.fence_create_info(flags=vk.FENCE_CREATE_SIGNALED_BIT)
    render_fences = tuple(hvk.create_fence(api, device, info) for _ in range(len(swapchain_images)))


def record_render_commands():

    # Render commands recording
    begin_info = hvk.command_buffer_begin_info()
    width, height = window.dimensions()

    render_pass_begin = hvk.render_pass_begin_info(
        render_pass = render_pass, framebuffer = 0,
        render_area = hvk.rect_2d(0, 0, width, height),
        clear_values = (
            hvk.clear_value(color=(0.0, 0.0, 0.0, 1.0)),
            hvk.clear_value(depth=1.0, stencil=0)
        )
    )

    for framebuffer, cmd in zip(framebuffers, cmd_draw):
        render_pass_begin.framebuffer = framebuffer

        hvk.begin_command_buffer(api, cmd, begin_info)
        hvk.begin_render_pass(api, cmd, render_pass_begin, vk.SUBPASS_CONTENTS_INLINE)

        hvk.bind_pipeline(api, cmd, pipeline, vk.PIPELINE_BIND_POINT_GRAPHICS)

        hvk.bind_index_buffer(api, cmd, mesh_buffer, mesh_indices['offset'], vk.INDEX_TYPE_UINT16)
        hvk.bind_vertex_buffers(api, cmd, (mesh_buffer, mesh_buffer), (mesh_positions['offset'], mesh_colors['offset']))

        hvk.draw_indexed(api, cmd, mesh_indices['count'])
        
        hvk.end_render_pass(api, cmd)
        hvk.end_command_buffer(api, cmd)


def render():
    # Render commands submitting

    # Get next image in core swapchain
    image_index, result = hvk.acquire_next_image(
        api, device, swapchain, semaphore = image_ready
    )
    
    # Wait until the rendering of the last scene is done
    fence = render_fences[image_index]
    hvk.wait_for_fences(api, device, (fence,))
    hvk.reset_fences(api, device, (fence,))

    # Start rendering on the next image
    submit_info = hvk.submit_info(
        wait_dst_stage_mask = (vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,),
        wait_semaphores = (image_ready,),
        signal_semaphores = (rendering_done,),
        command_buffers = (cmd_draw[image_index],)
    )

    hvk.queue_submit(api, render_queue.handle, (submit_info,), fence = fence)

    # Present the next image once rendering is done
    hvk.queue_present(api, render_queue.handle, hvk.present_info(
        swapchains = (swapchain,),
        image_indices = (image_index,),
        wait_semaphores = (rendering_done,)
    ))


def clean_resources():
    window.hide()
    hvk.device_wait_idle(api, device) 

    hvk.destroy_fence(api, device, staging_fence)
    hvk.destroy_command_pool(api, device, staging_pool)

    hvk.destroy_buffer(api, device, staging_mesh_buffer)
    hvk.free_memory(api, device, staging_mesh_memory)

    hvk.destroy_command_pool(api, device, drawing_pool)

    hvk.destroy_buffer(api, device, mesh_buffer)
    hvk.free_memory(api, device, mesh_memory)

    hvk.destroy_pipeline(api, device, pipeline)
    hvk.destroy_pipeline_cache(api, device, pipeline_cache)
    hvk.destroy_pipeline_layout(api, device, pipeline_layout)

    for m in shader_modules:
        hvk.destroy_shader_module(api, device, m)

    hvk.destroy_descriptor_set_layout(api, device, descriptor_set_layout)

    for fb in framebuffers:
        hvk.destroy_framebuffer(api, device, fb)

    hvk.destroy_render_pass(api, device, render_pass)

    hvk.destroy_semaphore(api, device, image_ready)
    hvk.destroy_semaphore(api, device, rendering_done)
    for f in render_fences:
        hvk.destroy_fence(api, device, f)

    hvk.destroy_image(api, device, depth_stencil)
    hvk.destroy_image_view(api, device, depth_view)
    hvk.free_memory(api, device, depth_alloc)

    for v in swapchain_image_views:
        hvk.destroy_image_view(api, device, v)

    hvk.destroy_swapchain(api, device, swapchain)
    hvk.destroy_device(api, device)
    hvk.destroy_surface(api, instance, surface)

    debugger.stop()
    hvk.destroy_instance(api, instance)

    window.destroy()

# Run setup
create_window()
create_instance()
create_debugger()
create_surface()

create_device()
get_queues()

create_staging_command()

create_swapchain()
setup_swapchain_image_views()
setup_swapchain_depth_stencil()

load_mesh()
mesh_to_staging()
mesh_to_device()

create_render_pass()
create_framebuffers()
create_shaders()
create_descriptor_set_layout()
create_pipeline_layout()
setup_vertex_input_binding()
create_pipeline()

create_render_resources()
record_render_commands()

window.show()

# Render loop
while not window.must_exit:
    window.translate_system_events()
    render()
    time.sleep(1/60)

clean_resources()
