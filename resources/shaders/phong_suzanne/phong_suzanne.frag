#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable

layout (location = 0) in vec3 inNorm;

layout (binding = 1) uniform LIGHT 
{
    vec4 reverseLightDirection;
} light;

layout (binding = 2) uniform MATERIAL 
{
    vec4 color;
    vec4 specular_shininess; // "rgb" contains specular / "a" contains shininess
} mat;


layout (location = 0) out vec4 outFragColor;

void main() 
{
    vec3 normal = normalize(inNorm);
    float diffuse = max(dot(normal, light.reverseLightDirection.xyz), 0.1);

    outFragColor = mat.color;
    outFragColor.rgb *= diffuse;
}