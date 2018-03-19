// Shader ported from https://github.com/KhronosGroup/glTF-WebGL-PBR

#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec2 inUV;
layout (location = 1) in vec2 inPosition;
layout (location = 2) in mat3 inTBN;

layout (binding = 1, packed) uniform LIGHT 
{
    vec4 reverseLightDirection;
    vec4 lightColor;
} light;

layout (binding = 2, packed) uniform PBR 
{
    vec4 baseColorFactor;
    vec2 metallicRoughnessValues;
} pbr;

layout (binding = 2) uniform sampler2DArray samplerPBR;

layout (location = 0) out vec4 outFragColor;

void main() 
{
    float perceptualRoughness = pbr.metallicRoughnessValues.y;
    float metallic = pbr.metallicRoughnessValues.x;

    outFragColor = vec4(1.0, 1.0, 1.0, 1.0);
}