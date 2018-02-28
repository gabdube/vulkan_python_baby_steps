#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inCol;

layout (location = 0) out vec3 fragColor;

void main() 
{
    fragColor = inCol;
	gl_Position = vec4(inPos.xyz, 1.0);
}
