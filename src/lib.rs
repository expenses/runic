// Disable the clippy lint about having too many function arguments
#![cfg_attr(feature = "cargo-clippy", allow(too_many_arguments))]

extern crate rusttype;
extern crate glium;
extern crate arrayvec;

use arrayvec::*;
use rusttype::*;
use rusttype::gpu_cache::*;
use glium::texture::*;
use glium::uniforms::*;
use glium::backend::*;
use glium::*;

use std::borrow::Cow;

/// The default vertex shader.
pub const VERT: &str = include_str!("shaders/shader.vert");
/// The default fragmentation shader.
pub const FRAG: &str = include_str!("shaders/shader.frag");
/// A shader for pixelated fonts.
pub const PIXELATED_VERT: &str = include_str!("shaders/shader_pixelated.vert");

/// Get the default program for rendering glyphs, which just renders them with a solid colour.
pub fn default_program<F: Facade>(display: &F) -> Result<Program, ProgramCreationError> {
    Program::from_source(display, VERT, FRAG, None)
}

pub fn pixelated_program<F: Facade>(display: &F) -> Result<Program, ProgramCreationError> {
    Program::from_source(display, PIXELATED_VERT, FRAG, None)
}

#[test]
fn compile_shaders() {
    use glium::glutin::*;
    use glium::backend::glutin::headless::*;
    let context = HeadlessRendererBuilder::new(1000, 1000).build().unwrap();
    let display = Headless::new(context).unwrap();
    default_program(&display).unwrap();
    pixelated_program(&display).unwrap();
}

fn screen_pos_to_opengl_pos(x: f32, y: f32, screen_width: f32, screen_height: f32) -> [f32; 2] {
    [
        (x / screen_width - 0.5) * 2.0,
        (1.0 - y / screen_height - 0.5) * 2.0
    ]
} 

fn screen_rect_to_opengl_rect(rect: rusttype::Rect<i32>, screen_width: f32, screen_height: f32) -> rusttype::Rect<f32> {
    rusttype::Rect {
        min: point(
            (rect.min.x as f32 / screen_width - 0.5) * 2.0,
            (1.0 - rect.min.y as f32 / screen_height - 0.5) * 2.0
        ),
        max: point(
            (rect.max.x as f32 / screen_width - 0.5) * 2.0,
            (1.0 - rect.max.y as f32 / screen_height - 0.5) * 2.0
        )
    }
}

/// A vertex for rendering.
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    pub in_pos: [f32; 2],
    pub in_uv: [f32; 2],
}

implement_vertex!(Vertex, in_pos, in_uv);

/// A Glyph Cache for caching the textures of font(s).
pub struct GlyphCache {
    cache: Cache<'static>,
    cache_tex: Texture2d,
}

impl GlyphCache {
    /// Create a new glyph cache.
    ///
    /// Will fail if the texture fails to be created for whatever reason.
    pub fn new(display: &Display) -> Result<Self, Error> {
        let dpi = display.gl_window().get_hidpi_factor();
        let (cache_width, cache_height) = ((512.0 * dpi) as u32, (512.0 * dpi) as u32);
        let cache = Cache::builder().dimensions(cache_width, cache_height).build();

        let cache_tex = Texture2d::with_format(
            display,
            RawImage2d {
                data: Cow::Owned(vec![128u8; cache_width as usize * cache_height as usize]),
                width: cache_width,
                height: cache_height,
                format: ClientFormat::U8,
            },
            UncompressedFloatFormat::U8,
            MipmapsOption::NoMipmap,
        ).map_err(Error::TextureCreation)?;

        Ok(Self {
            cache, cache_tex
        })
    }

    pub fn texture(&self) -> &Texture2d {
        &self.cache_tex
    }

    fn update_cache<'a>(&mut self, text: &'a str, scale: f32, dpi: f32, font: &'a Font, font_id: usize, origin: [f32; 2], pixelated: bool) -> Result<impl Iterator<Item=PositionedGlyph<'static>> + 'a, Error> {
        let glyphs = layout_glyphs(text, scale, dpi, origin, font, pixelated);

        for glyph in glyphs.clone() {
            self.cache.queue_glyph(font_id, glyph.clone());
        }

        let cache_tex = &self.cache_tex;

        self.cache.cache_queued(|rect, data| {
            // If we're using a pixelated font we need to round the coverage values
            let data = if pixelated {
                Cow::Owned(data.into_iter().map(|value| if *value > 128 {255} else {0}).collect())
            } else {
                Cow::Borrowed(data)
            };

            cache_tex.main_level().write(
                glium::Rect {
                    left: rect.min.x,
                    bottom: rect.min.y,
                    width: rect.width(),
                    height: rect.height(),
                },
                RawImage2d {
                    data,
                    width: rect.width(),
                    height: rect.height(),
                    format: glium::texture::ClientFormat::U8,
                }
            );
        }).map_err(Error::CacheWrite)?;

        Ok(glyphs)
    }

    /// Get the vertices for a piece of text renderered via a font.
    ///
    /// Most of the arguments are fairly self explanatory, but:
    ///
    /// `scale` is the logical font size (as opposed to the physical font size which depends on dpi).
    ///
    /// `font_id` is the ID of the font in the cache. It is highly recommended that you only cache one font in a glyph cache, so this will almost always be `0`.
    ///
    /// `origin`: The button left corner of the rendered text in logical size. Uses scaled opengl coordinates so `(0, 0)` is the bottom left corner of the screen and `(width, height)` is the top right.
    ///
    /// `pixelated`: Whether the font rendered in a pixelated way or not. This will round coverage values etc to make the font look nicer.
    ///
    /// **Tip**: This method for rendering pixelated fonts only really works well when the pixels are at a 1:1 ratio with the screen. I recommend using a shader to scale the rendered text up instead of increasing the font size.
    pub fn get_vertices(&mut self, text: &str, origin: [f32; 2], scale: f32, font: &Font, font_id: usize, pixelated: bool, display: &Display) -> Result<Vec<Vertex>, Error> {
        let dpi = display.gl_window().get_hidpi_factor() as f32;
        // Scale the origin and scale by the dpi to get the physical versions (e.g. in pixels)

        let glyphs = self.update_cache(text, scale, dpi, font, font_id, origin, pixelated)?;

        let (screen_width, screen_height) = {
            let (screen_width, screen_height) = display.get_framebuffer_dimensions();
            (screen_width as f32, screen_height as f32)
        };

        let vertices = glyphs
            .into_iter()
            .filter_map(|glyph| self.cache.rect_for(font_id, &glyph).ok())
            .filter_map(|rects| rects)
            .flat_map(|(uv_rect, screen_rect)| {
                // Scale down the screen rectangle to opengl coordinates.
                // We don't _need_ to do this here, and could do it in opengl instead, but this way makes writing custom shaders easier
                let gl_rect = screen_rect_to_opengl_rect(screen_rect, screen_width, screen_height);

                ArrayVec::from([
                    Vertex {
                        in_pos: [gl_rect.min.x, gl_rect.max.y],
                        in_uv: [uv_rect.min.x, uv_rect.max.y]
                    },
                    Vertex {
                        in_pos: [gl_rect.min.x, gl_rect.min.y],
                        in_uv: [uv_rect.min.x, uv_rect.min.y]
                    },
                    Vertex {
                        in_pos: [gl_rect.max.x, gl_rect.min.y],
                        in_uv: [uv_rect.max.x, uv_rect.min.y]
                    },
                    Vertex {
                        in_pos: [gl_rect.max.x, gl_rect.min.y],
                        in_uv: [uv_rect.max.x, uv_rect.min.y],
                    },
                    Vertex {
                        in_pos: [gl_rect.max.x, gl_rect.max.y],
                        in_uv: [uv_rect.max.x, uv_rect.max.y],
                    },
                    Vertex {
                        in_pos: [gl_rect.min.x, gl_rect.max.y],
                        in_uv: [uv_rect.min.x, uv_rect.max.y],
                    }
                ])
            })
            .collect();

        Ok(vertices)
    }

    /// A helper function to render text out to a target using a shader pipeline.
    ///
    /// If you want more control over rendering in regards to uniforms and the like, you can [`get_vertices`] to get the vertices
    /// for rendering.
    ///
    /// See [`get_vertices`] for infomation on the arguments.
    ///
    /// [`get_vertices`]: #method.get_vertices
    pub fn render<S: Surface>(&mut self, text: &str, origin: [f32; 2], scale: f32, colour: [f32; 4], font: &Font, font_id: usize, target: &mut S, display: &Display, program: &Program) -> Result<(), Error> {
        let vertices = self.get_vertices(text, origin, scale, font, font_id, false, display)?;

        let uniforms = uniform! {
            sampler: Sampler::new(&self.cache_tex),
            colour: colour
        };

        let vertex_buffer = VertexBuffer::new(display, &vertices).map_err(Error::BufferCreation)?;

        target.draw(
            &vertex_buffer,
            glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
            program,
            &uniforms,
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                ..Default::default()
            }
        ).map_err(Error::Draw)?;

        Ok(())
    }

    pub fn render_pixelated<S: Surface>(&mut self, text: &str, origin: [f32; 2], font_size: f32, scale: f32, colour: [f32; 4], font: &Font, font_id: usize, target: &mut S, display: &Display, program: &Program) -> Result<(), Error> {
        let dpi = display.gl_window().get_hidpi_factor() as f32;

        let vertices = self.get_vertices(text, origin, font_size, font, font_id, true, display)?;

        let (screen_width, screen_height) = {
            let (screen_width, screen_height) = display.get_framebuffer_dimensions();
            (screen_width as f32, screen_height as f32)
        };

        let uniforms = uniform! {
            sampler: Sampler::new(&self.cache_tex).magnify_filter(MagnifySamplerFilter::Nearest),
            scale: scale,
            origin: screen_pos_to_opengl_pos(origin[0] * dpi, origin[1] * dpi, screen_width, screen_height),
            colour: colour,
        };

        let vertex_buffer = VertexBuffer::new(display, &vertices).map_err(Error::BufferCreation)?;

        target.draw(
            &vertex_buffer,
            glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList),
            program,
            &uniforms,
            &glium::DrawParameters {
                blend: glium::Blend::alpha_blending(),
                ..Default::default()
            }
        ).map_err(Error::Draw)?;

        Ok(())
    }
}

fn layout_glyphs<'a>(text: &'a str, scale: f32, dpi: f32, start: [f32; 2], font: &'a Font, pixelated: bool) -> impl Iterator<Item=PositionedGlyph<'static>> + Clone + 'a {
    let scale = Scale::uniform(scale * dpi);
    let start = point(start[0] * dpi, start[1] * dpi + font.v_metrics(scale).ascent);

    font.glyphs_for(text.chars())
        .scan((None, 0.0), move |&mut (ref mut last, ref mut x), g| {
            let g = g.scaled(scale);
            let mut w = g.h_metrics().advance_width
                + last.map(|last| font.pair_kerning(scale, last, g.id())).unwrap_or(0.0);
            
            // If we're using a pixelated font we need to round the width of the glyph to prevent errors
            if pixelated {
                w = w.round()
            }
            
            let next = g.positioned(start + vector(*x, 0.0));

            *last = Some(next.id());
            *x += w;
            Some(next)
        })
        .map(|glyph| glyph.standalone())
}

/// Get the logical size of a piece of rendered text.
pub fn rendered_size(text: &str, scale: f32, font: &Font, pixelated: bool, display: &Display) -> (f32, f32) {
    let dpi = display.gl_window().get_hidpi_factor() as f32;

    let height = font.v_metrics(Scale::uniform(scale)).ascent;

    let width = layout_glyphs(text, scale, dpi, [0.0, 0.0], font, pixelated)
        .filter_map(|glyph| glyph.pixel_bounding_box())
        .fold(0.0_f32, |width, bbox| width.max(bbox.max.x as f32 / dpi));

    (width, height)
}

/// A font with a contained glyph cache for ease of use.
pub struct CachedFont<'a> {
    font: Font<'a>,
    cache: GlyphCache
}

impl<'a> CachedFont<'a> {
    /// Setup the struct and create the glyph cache.
    pub fn new(font: Font<'a>, display: &Display) -> Result<Self, Error> {
        Ok(Self {
            font,
            cache: GlyphCache::new(display)?
        })
    }

    pub fn from_bytes(bytes: &'a [u8], display: &Display) -> Result<Self, Error> {
        Self::new(
            Font::from_bytes(bytes).map_err(Error::Font)?,
            display
        )
    }

    pub fn inner(&self) -> &Font<'a> {
        &self.font
    }

    pub fn cache(&self) -> &GlyphCache {
        &self.cache
    }

    /// Render the font onto a target via shaders.
    ///
    /// See [`GlyphCache::get_vertices`] for infomation on the arguments.
    ///
    /// [`GlyphCache::get_vertices`]: struct.GlyphCache.html#method.get_vertices
    pub fn render<S: Surface>(&mut self, text: &str, origin: [f32; 2], scale: f32, colour: [f32; 4], target: &mut S, display: &Display, program: &Program) -> Result<(), Error> {
        self.cache.render(text, origin, scale, colour, &self.font, 0, target, display, program)
    }

    pub fn render_pixelated<S: Surface>(&mut self, text: &str, origin: [f32; 2], font_size: f32, scale: f32, colour: [f32; 4], target: &mut S, display: &Display, program: &Program) -> Result<(), Error> {
        self.cache.render_pixelated(text, origin, font_size, scale, colour, &self.font, 0, target, display, program)
    }

    pub fn get_vertices(&mut self, text: &str, origin: [f32; 2], scale: f32, pixelated: bool, display: &Display) -> Result<Vec<Vertex>, Error> {
        self.cache.get_vertices(text, origin, scale, &self.font, 0, pixelated, display)
    }

    pub fn rendered_scale(&self, text: &str, scale: f32, pixelated: bool, display: &Display) -> (f32, f32) {
        rendered_size(text, scale, &self.font, pixelated, display)
    }
}

#[derive(Debug)]
/// All the errors that can occur.
pub enum Error {
    CacheWrite(CacheWriteErr),
    BufferCreation(vertex::BufferCreationError),
    Draw(DrawError),
    Font(rusttype::Error),
    TextureCreation(TextureCreationError)
}