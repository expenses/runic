use *;
use {Vertex, Error};

/// A Glyph Cache for caching the textures of fonts.
pub struct GlyphCache {
    cache: Cache<'static>,
    cache_tex: Texture2d,
}

impl GlyphCache {
    /// Create a new glyph cache.
    ///
    /// Will fail if the texture fails to be created for whatever reason.
    pub fn new<D: Display>(display: &D) -> Result<Self, Error> {
        let dpi = display.dpi_factor();

        // Create the cache
        let (cache_width, cache_height) = ((512.0 * dpi) as u32, (512.0 * dpi) as u32);
        let cache = Cache::builder().dimensions(cache_width, cache_height).build();

        // Create the cache texture
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

    fn update_cache<'a>(&mut self, text: &'a str, scale: f32, dpi: f32, font: &'a Font, font_id: usize, pixelated: bool) -> Result<impl Iterator<Item=PositionedGlyph<'static>> + 'a, Error> {
        // Get an iterator of laid out glyphs
        let glyphs = layout_glyphs(text, scale, dpi, font, pixelated);

        // queue up glyphs to cache
        for glyph in glyphs.clone() {
            self.cache.queue_glyph(font_id, glyph.clone());
        }

        // Cache all the queued glyphs

        let cache_tex = &self.cache_tex;

        self.cache.cache_queued(|rect, data| {
            // If we're using a pixelated font we need to round the coverage values
            let data = if pixelated {
                Cow::Owned(data.into_iter().map(|value| if *value > 128 {255} else {0}).collect())
            } else {
                Cow::Borrowed(data)
            };

            // Write the data to the texture
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
    pub fn get_vertices<D: Display>(&mut self, text: &str, origin: [f32; 2], scale: f32, font: &Font, font_id: usize, pixelated: bool, display: &D) -> Result<Vec<Vertex>, Error> {
        let dpi = display.dpi_factor();
        // Scale the origin and scale by the dpi to get the physical versions (e.g. in pixels)

        // Update the cache and get the glyph iterator
        let glyphs = self.update_cache(text, scale, dpi, font, font_id, pixelated)?;

        let (screen_width, screen_height) = {
            let (screen_width, screen_height) = display.framebuffer_dimensions();
            (screen_width as f32, screen_height as f32)
        };

        // Create a list of the vertices of glyphs in the cache texture (split into two triangles so we don't need to use and index buffer as well)
        let vertices = glyphs
            .into_iter()
            .filter_map(|glyph| self.cache.rect_for(font_id, &glyph).ok())
            .filter_map(|rects| rects)
            .flat_map(|(uv_rect, screen_rect)| {
                // Scale down the screen rectangle to opengl coordinates.
                // We don't _need_ to do this here, and could do it in opengl instead, but this makes writing custom shaders easier
                let gl_rect = screen_rect_to_opengl_rect(screen_rect, screen_width, screen_height, origin);

                let a = Vertex {
                    in_pos: [gl_rect.min.x, gl_rect.min.y],
                    in_uv: [uv_rect.min.x, uv_rect.min.y]
                };

                let b = Vertex {
                    in_pos: [gl_rect.min.x, gl_rect.max.y],
                    in_uv: [uv_rect.min.x, uv_rect.max.y]
                };

                let c = Vertex {
                    in_pos: [gl_rect.max.x, gl_rect.min.y],
                    in_uv: [uv_rect.max.x, uv_rect.min.y]
                };

                let d = Vertex {
                    in_pos: [gl_rect.max.x, gl_rect.max.y],
                    in_uv: [uv_rect.max.x, uv_rect.max.y],
                };

                ArrayVec::from([a, b, c, b, c, d])
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
    pub fn render<S: Surface, D: Display>(&mut self, text: &str, origin: [f32; 2], scale: f32, colour: [f32; 4], font: &Font, font_id: usize, target: &mut S, display: &D, program: &Program) -> Result<(), Error> {
        // Get the vertices and create a vertex buffer
        let vertices = self.get_vertices(text, origin, scale, font, font_id, false, display)?;
        let vertex_buffer = VertexBuffer::new(display, &vertices).map_err(Error::BufferCreation)?;

        // Create the uniforms
        let uniforms = uniform! {
            sampler: Sampler::new(&self.cache_tex),
            colour: colour
        };

        // Draw to the target!
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

    pub fn render_pixelated<S: Surface, D: Display>(&mut self, text: &str, mut origin: [f32; 2], font_size: f32, scale: f32, colour: [f32; 4], font: &Font, font_id: usize, target: &mut S, display: &D, program: &Program) -> Result<(), Error> {
        let dpi = display.dpi_factor();
      
        // Multiply the origin by the dpi because of reasons to do with the `layout_glyphs` thing.
        origin[0] *= dpi;
        origin[1] *= dpi;

        let vertices = self.get_vertices(text, origin, font_size, font, font_id, true, display)?;

        let vertex_buffer = VertexBuffer::new(display, &vertices).map_err(Error::BufferCreation)?;

        let (screen_width, screen_height) = {
            let (screen_width, screen_height) = display.framebuffer_dimensions();
            (screen_width as f32, screen_height as f32)
        };

        let uniforms = uniform! {
            // Set the magnify filter to nearest as the shader is going to upscale the text by the scale
            sampler: Sampler::new(&self.cache_tex).magnify_filter(MagnifySamplerFilter::Nearest),
            // Increase the scale by dpi because we lower the at the start of `layout_glyphs`
            scale: scale * dpi,
            // Translate the origin down to opengl coordinates
            origin: screen_pos_to_opengl_pos(origin, screen_width, screen_height),
            colour: colour,
        };

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


// todo: maybe use an iterator struct for this
pub fn layout_glyphs<'a>(text: &'a str, scale: f32, mut dpi: f32, font: &'a Font, pixelated: bool) -> impl Iterator<Item=PositionedGlyph<'static>> + Clone + 'a {
    // As we're rendering to a texture on the gpu that knows nothing about dpi, we want to render pixelated fonts at the regular dpi level,
    // as rendering them too large can make them look different between high dpi screens and regular dpi screens.
    if pixelated {
        dpi = 1.0;
    }

    let scale = Scale::uniform(scale * dpi);
    let start = point(0.0, font.v_metrics(scale).ascent);

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