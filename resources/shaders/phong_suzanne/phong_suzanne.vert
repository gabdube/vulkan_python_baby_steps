#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNorm;

layout (location = 0) out vec3 outFragPos;
layout (location = 1) out vec3 outNorm;

layout (binding = 0) uniform UBO 
{
    mat4 proj;
    mat4 view;
    mat4 model;
    mat4 normal;
} ubo;


void main() 
{
    outFragPos = vec3(ubo.model * vec4(inPos, 1.0));
    outNorm = mat3(ubo.normal) * inNorm;

    gl_Position = ubo.proj * ubo.view * vec4(outFragPos, 1.0);
}
