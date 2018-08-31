#version 140

in vec2 in_pos;
in vec2 in_uv;

out vec2 out_uv;

uniform vec2 origin;
uniform float scale;

void main() {
    // Scale the position around the origin by `scale`
    vec2 pos = origin + (in_pos - origin) * scale;
    gl_Position = vec4(pos, 0.0, 1.0);
    out_uv = in_uv;
}