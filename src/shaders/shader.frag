#version 140

uniform sampler2D sampler;
uniform vec4 colour;

in vec2 out_uv;

out vec4 texel;

void main() {
    texel = colour * vec4(1.0, 1.0, 1.0, texture(sampler, out_uv).r);
}