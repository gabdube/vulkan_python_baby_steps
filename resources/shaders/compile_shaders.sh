source /home/gdube/Documents/other/VulkanSDK/1.0.68.0/setup-env.sh

glslangValidator -V ./shaded_cube/shaded_cube.vert -o ./shaded_cube/shaded_cube.vert.spv
glslangValidator -V ./shaded_cube/shaded_cube.frag -o ./shaded_cube/shaded_cube.frag.spv
