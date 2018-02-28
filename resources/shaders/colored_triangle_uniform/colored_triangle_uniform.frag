#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (binding = 0) uniform COLOR 
{
    vec4 value;
} color;


layout (location = 0) out vec4 outFragColor;

void main() 
{
  outFragColor = color.value;
}