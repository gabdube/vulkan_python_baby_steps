source /home/gdube/Documents/other/VulkanSDK/1.0.68.0/setup-env.sh

glslangValidator -V ./shaded_cube/shaded_cube.vert -o ./shaded_cube/shaded_cube.vert.spv
glslangValidator -V ./shaded_cube/shaded_cube.frag -o ./shaded_cube/shaded_cube.frag.spv

glslangValidator -V ./simple_triangle/simple_triangle.vert -o ./simple_triangle/simple_triangle.vert.spv
glslangValidator -V ./simple_triangle/simple_triangle.frag -o ./simple_triangle/simple_triangle.frag.spv

glslangValidator -V ./colored_triangle_attribute/colored_triangle_attribute.vert -o ./colored_triangle_attribute/colored_triangle_attribute.vert.spv
glslangValidator -V ./colored_triangle_attribute/colored_triangle_attribute.frag -o ./colored_triangle_attribute/colored_triangle_attribute.frag.spv

glslangValidator -V ./colored_triangle_uniform/colored_triangle_uniform.vert -o ./colored_triangle_uniform/colored_triangle_uniform.vert.spv
glslangValidator -V ./colored_triangle_uniform/colored_triangle_uniform.frag -o ./colored_triangle_uniform/colored_triangle_uniform.frag.spv

glslangValidator -V ./textured_cube/textured_cube.vert -o ./textured_cube/textured_cube.vert.spv
glslangValidator -V ./textured_cube/textured_cube.frag -o ./textured_cube/textured_cube.frag.spv

glslangValidator -V ./compute_noise/compute_noise.vert -o ./compute_noise/compute_noise.vert.spv
glslangValidator -V ./compute_noise/compute_noise.frag -o ./compute_noise/compute_noise.frag.spv
