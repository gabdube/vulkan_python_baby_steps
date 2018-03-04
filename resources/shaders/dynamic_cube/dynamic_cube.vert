#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNorm;

layout (binding = 0) uniform UBO 
{
    mat4 proj;
    mat4 view;
    mat4 model;
} ubo;

void main() 
{
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);
}
