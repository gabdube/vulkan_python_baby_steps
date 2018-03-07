source /home/gdube/Documents/other/VulkanSDK/1.0.68.0/setup-env.sh

glslangValidator -V ./dynamic_cube/dynamic_cube.vert -o ./dynamic_cube/dynamic_cube.vert.spv
glslangValidator -V ./dynamic_cube/dynamic_cube.frag -o ./dynamic_cube/dynamic_cube.frag.spv
