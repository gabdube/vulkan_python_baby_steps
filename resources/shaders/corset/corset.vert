// Shader ported from https://github.com/KhronosGroup/glTF-WebGL-PBR

#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inPos;
layout (location = 1) in vec3 inNorm;
layout (location = 2) in vec2 inUV;
layout (location = 3) in vec4 inTangent;

layout (binding = 0) uniform UBO 
{
    mat4 proj;
    mat4 view;
    mat4 model;
    mat4 normal;
} ubo;

layout (location = 0) out vec2 outUV;
layout (location = 1) out vec3 outPosition;
layout (location = 2) out mat3 outTBN;

void main() 
{
    vec4 pos = ubo.model * vec4(inPos.xyz, 1.0);
    outPosition = vec3(pos.xyz) / pos.w; 

    vec3 normalW = normalize(vec3(ubo.normal * vec4(inNorm.xyz, 0.0)));
    vec3 tangentW = normalize(vec3(ubo.model * vec4(inTangent.xyz, 0.0)));
    vec3 bitangentW = cross(normalW, tangentW) * inTangent.w;
    outTBN = mat3(tangentW, bitangentW, normalW);

    outUV = inUV;

    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPos.xyz, 1.0);
}
