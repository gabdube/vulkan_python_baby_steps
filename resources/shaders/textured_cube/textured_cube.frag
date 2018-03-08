#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inNorm;
layout (location = 1) in vec2 inUV;

layout (binding = 1) uniform LIGHT 
{
    vec3 reverseLightDirection;
} light;

layout (binding = 2) uniform sampler2D samplerColor;

layout (location = 0) out vec4 outFragColor;

void main() 
{
  vec3 normal = normalize(inNorm);
  float light_v = max(dot(normal, light.reverseLightDirection.rgb), 0.1);

  outFragColor = texture(samplerColor, inUV, 0.0);
  outFragColor.rgb *= light_v;
}