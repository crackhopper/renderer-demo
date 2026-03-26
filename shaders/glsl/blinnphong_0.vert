#version 450

// DEBUG: absolute minimum vertex shader - hardcoded triangle, zero dependencies
void main() {
    vec2 positions[3] = vec2[](
        vec2(-0.5, -0.5),
        vec2( 0.5, -0.5),
        vec2( 0.0,  0.5)
    );
    gl_Position = vec4(positions[gl_VertexIndex % 3], 0.0, 1.0);
}
