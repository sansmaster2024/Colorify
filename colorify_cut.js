// No unsandbox check -> error!!!!!!!!!!!!!!!!

(function (Scratch) {
    const fragShader = "precision mediump float;\nuniform float u_tileX;\nuniform float u_tileY;\nuniform float u_tintR;\nuniform float u_tintG;\nuniform float u_tintB;\nuniform float u_tintA;\n#ifdef DRAW_MODE_silhouette\nuniform vec4 u_silhouetteColor;\n#else // DRAW_MODE_silhouette\n# ifdef ENABLE_color\nuniform float u_color;\n# endif // ENABLE_color\n# ifdef ENABLE_brightness\nuniform float u_brightness;\n# endif // ENABLE_brightness\n#endif // DRAW_MODE_silhouette\n#ifdef DRAW_MODE_colorMask\nuniform vec3 u_colorMask;\nuniform float u_colorMaskTolerance;\n#endif // DRAW_MODE_colorMask\n#ifdef ENABLE_fisheye\nuniform float u_fisheye;\n#endif // ENABLE_fisheye\n#ifdef ENABLE_whirl\nuniform float u_whirl;\n#endif // ENABLE_whirl\n#ifdef ENABLE_pixelate\nuniform float u_pixelate;\nuniform vec2 u_skinSize;\n#endif // ENABLE_pixelate\n#ifdef ENABLE_mosaic\nuniform float u_mosaic;\n#endif // ENABLE_mosaic\n#ifdef ENABLE_ghost\nuniform float u_ghost;\n#endif // ENABLE_ghost\n#ifdef DRAW_MODE_line\nvarying vec4 v_lineColor;\nvarying float v_lineThickness;\nvarying float v_lineLength;\n#endif // DRAW_MODE_line\n#ifdef DRAW_MODE_background\nuniform vec4 u_backgroundColor;\n#endif // DRAW_MODE_background\nuniform sampler2D u_skin;\n#ifndef DRAW_MODE_background\nvarying vec2 v_texCoord;\n#endif\n// Add this to divisors to prevent division by 0, which results in NaNs propagating through calculations.\n// Smaller values can cause problems on some mobile devices.\nconst float epsilon = 1e-3;\n#if !defined(DRAW_MODE_silhouette) && (defined(ENABLE_color))\n// Branchless color conversions based on code from:\n// http://www.chilliant.com/rgb2hsv.html by Ian Taylor\n// Based in part on work by Sam Hocevar and Emil Persson\n// See also: https://en.wikipedia.org/wiki/HSL_and_HSV#Formal_derivation\n// Convert an RGB color to Hue, Saturation, and Value.\n// All components of input and output are expected to be in the [0,1] range.\nvec3 convertRGB2HSV(vec3 rgb)\n{\n\t// Hue calculation has 3 cases, depending on which RGB component is largest, and one of those cases involves a \"mod\"\n\t// operation. In order to avoid that \"mod\" we split the M==R case in two: one for G<B and one for B>G. The B>G case\n\t// will be calculated in the negative and fed through abs() in the hue calculation at the end.\n\t// See also: https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma\n\tconst vec4 hueOffsets = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);\n\t// temp1.xy = sort B & G (largest first)\n\t// temp1.z = the hue offset we'll use if it turns out that R is the largest component (M==R)\n\t// temp1.w = the hue offset we'll use if it turns out that R is not the largest component (M==G or M==B)\n\tvec4 temp1 = rgb.b > rgb.g ? vec4(rgb.bg, hueOffsets.wz) : vec4(rgb.gb, hueOffsets.xy);\n\t// temp2.x = the largest component of RGB (\"M\" / \"Max\")\n\t// temp2.yw = the smaller components of RGB, ordered for the hue calculation (not necessarily sorted by magnitude!)\n\t// temp2.z = the hue offset we'll use in the hue calculation\n\tvec4 temp2 = rgb.r > temp1.x ? vec4(rgb.r, temp1.yzx) : vec4(temp1.xyw, rgb.r);\n\t// m = the smallest component of RGB (\"min\")\n\tfloat m = min(temp2.y, temp2.w);\n\t// Chroma = M - m\n\tfloat C = temp2.x - m;\n\t// Value = M\n\tfloat V = temp2.x;\n\treturn vec3(\n\t\tabs(temp2.z + (temp2.w - temp2.y) / (6.0 * C + epsilon)), // Hue\n\t\tC / (temp2.x + epsilon), // Saturation\n\t\tV); // Value\n}\nvec3 convertHue2RGB(float hue)\n{\n\tfloat r = abs(hue * 6.0 - 3.0) - 1.0;\n\tfloat g = 2.0 - abs(hue * 6.0 - 2.0);\n\tfloat b = 2.0 - abs(hue * 6.0 - 4.0);\n\treturn clamp(vec3(r, g, b), 0.0, 1.0);\n}\nvec3 convertHSV2RGB(vec3 hsv)\n{\n\tvec3 rgb = convertHue2RGB(hsv.x);\n\tfloat c = hsv.z * hsv.y;\n\treturn rgb * c + hsv.z - c;\n}\n#endif // !defined(DRAW_MODE_silhouette) && (defined(ENABLE_color))\nconst vec2 kCenter = vec2(0.5, 0.5);\nvoid main()\n{\n\t#if !(defined(DRAW_MODE_line) || defined(DRAW_MODE_background))\n\tvec2 texcoord0 = v_texCoord;\n\t#ifdef ENABLE_mosaic\n\ttexcoord0 = fract(u_mosaic * texcoord0);\n\t#endif // ENABLE_mosaic\n\t#ifdef ENABLE_pixelate\n\t{\n\t\t// TODO: clean up \"pixel\" edges\n\t\tvec2 pixelTexelSize = u_skinSize / u_pixelate;\n\t\ttexcoord0 = (floor(texcoord0 * pixelTexelSize) + kCenter) / pixelTexelSize;\n\t}\n\t#endif // ENABLE_pixelate\n\t#ifdef ENABLE_whirl\n\t{\n\t\tconst float kRadius = 0.5;\n\t\tvec2 offset = texcoord0 - kCenter;\n\t\tfloat offsetMagnitude = length(offset);\n\t\tfloat whirlFactor = max(1.0 - (offsetMagnitude / kRadius), 0.0);\n\t\tfloat whirlActual = u_whirl * whirlFactor * whirlFactor;\n\t\tfloat sinWhirl = sin(whirlActual);\n\t\tfloat cosWhirl = cos(whirlActual);\n\t\tmat2 rotationMatrix = mat2(\n\t\t\tcosWhirl, -sinWhirl,\n\t\t\tsinWhirl, cosWhirl\n\t\t);\n\t\ttexcoord0 = rotationMatrix * offset + kCenter;\n\t}\n\t#endif // ENABLE_whirl\n\t#ifdef ENABLE_fisheye\n\t{\n\t\tvec2 vec = (texcoord0 - kCenter) / kCenter;\n\t\tfloat vecLength = length(vec);\n\t\tfloat r = pow(min(vecLength, 1.0), u_fisheye) * max(1.0, vecLength);\n\t\tvec2 unit = vec / vecLength;\n\t\ttexcoord0 = kCenter + r * unit * kCenter;\n\t}\n\t#endif // ENABLE_fisheye\n\ttexcoord0 = vec2(mod(texcoord0.x*u_tileX, 1.0), mod(texcoord0.y*u_tileY, 1.0));\n\tgl_FragColor = texture2D(u_skin, texcoord0);\n\t#if defined(ENABLE_color) || defined(ENABLE_brightness)\n\t// Divide premultiplied alpha values for proper color processing\n\t// Add epsilon to avoid dividing by 0 for fully transparent pixels\n\tgl_FragColor.rgb = clamp(gl_FragColor.rgb / (gl_FragColor.a + epsilon), 0.0, 1.0);\n\t#ifdef ENABLE_color\n\t{\n\t\tvec3 hsv = convertRGB2HSV(gl_FragColor.xyz);\n\t\t// this code forces grayscale values to be slightly saturated\n\t\t// so that some slight change of hue will be visible\n\t\tconst float minLightness = 0.0;\n\t\tconst float minSaturation = 0.0;\n\t\tif (hsv.z < minLightness) hsv = vec3(0.0, 1.0, minLightness);\n\t\telse if (hsv.y < minSaturation) hsv = vec3(0.0, minSaturation, hsv.z);\n\t\thsv.x = mod(hsv.x + u_color, 1.0);\n\t\tif (hsv.x < 0.0) hsv.x += 1.0;\n\t\tgl_FragColor.rgb = convertHSV2RGB(hsv);\n\t}\n\t#endif // ENABLE_color\n\t#ifdef ENABLE_brightness\n\tgl_FragColor.rgb = clamp(gl_FragColor.rgb + vec3(u_brightness), vec3(0), vec3(1));\n\t#endif // ENABLE_brightness\n\t// Re-multiply color values\n\tgl_FragColor.rgb *= gl_FragColor.a + epsilon;\n\t#endif // defined(ENABLE_color) || defined(ENABLE_brightness)\n\t#ifdef ENABLE_ghost\n\tgl_FragColor *= u_ghost;\n\t#endif // ENABLE_ghost\n\t#ifdef DRAW_MODE_silhouette\n\t// Discard fully transparent pixels for stencil test\n\tif (gl_FragColor.a == 0.0) {\n\t\tdiscard;\n\t}\n\t// switch to u_silhouetteColor only AFTER the alpha test\n\tgl_FragColor = u_silhouetteColor;\n\t#else // DRAW_MODE_silhouette\n\t#ifdef DRAW_MODE_colorMask\n\tvec3 maskDistance = abs(gl_FragColor.rgb - u_colorMask);\n\tvec3 colorMaskTolerance = vec3(u_colorMaskTolerance, u_colorMaskTolerance, u_colorMaskTolerance);\n\tif (any(greaterThan(maskDistance, colorMaskTolerance)))\n\t{\n\t\tdiscard;\n\t}\n\t#endif // DRAW_MODE_colorMask\n\t#endif // DRAW_MODE_silhouette\n\t#ifdef DRAW_MODE_straightAlpha\n\t// Un-premultiply alpha.\n\tgl_FragColor.rgb /= gl_FragColor.a + epsilon;\n\t#endif\n\t#endif // !(defined(DRAW_MODE_line) || defined(DRAW_MODE_background))\n\t#ifdef DRAW_MODE_line\n\t// Maaaaagic antialiased-line-with-round-caps shader.\n\t// \"along-the-lineness\". This increases parallel to the line.\n\t// It goes from negative before the start point, to 0.5 through the start to the end, then ramps up again\n\t// past the end point.\n\tfloat d = ((v_texCoord.x - clamp(v_texCoord.x, 0.0, v_lineLength)) * 0.5) + 0.5;\n\t// Distance from (0.5, 0.5) to (d, the perpendicular coordinate). When we're in the middle of the line,\n\t// d will be 0.5, so the distance will be 0 at points close to the line and will grow at points further from it.\n\t// For the \"caps\", d will ramp down/up, giving us rounding.\n\t// See https://www.youtube.com/watch?v=PMltMdi1Wzg for a rough outline of the technique used to round the lines.\n\tfloat line = distance(vec2(0.5), vec2(d, v_texCoord.y)) * 2.0;\n\t// Expand out the line by its thickness.\n\tline -= ((v_lineThickness - 1.0) * 0.5);\n\t// Because \"distance to the center of the line\" decreases the closer we get to the line, but we want more opacity\n\t// the closer we are to the line, invert it.\n\tgl_FragColor = v_lineColor * clamp(1.0 - line, 0.0, 1.0);\n\t#endif // DRAW_MODE_line\n\t#ifdef DRAW_MODE_background\n\tgl_FragColor = u_backgroundColor;\n\t#endif\n\t\n\tgl_FragColor *= vec4(1.0-u_tintR, 1.0-u_tintG, 1.0-u_tintB, 1.0-u_tintA);\n}\n\t";
    const vsShader = "precision mediump float;\n#ifdef DRAW_MODE_line\nuniform vec2 u_stageSize;\nattribute vec2 a_lineThicknessAndLength;\nattribute vec4 a_penPoints;\nattribute vec4 a_lineColor;\nvarying vec4 v_lineColor;\nvarying float v_lineThickness;\nvarying float v_lineLength;\nvarying vec4 v_penPoints;\n// Add this to divisors to prevent division by 0, which results in NaNs propagating through calculations.\n// Smaller values can cause problems on some mobile devices.\nconst float epsilon = 1e-3;\n#endif\n#if !(defined(DRAW_MODE_line) || defined(DRAW_MODE_background))\nuniform mat4 u_projectionMatrix;\nuniform mat4 u_modelMatrix;\nattribute vec2 a_texCoord;\n#endif\nattribute vec2 a_position;\nvarying vec2 v_texCoord;\nvoid main() {\n\t#ifdef DRAW_MODE_line\n\t// Calculate a rotated (\"tight\") bounding box around the two pen points.\n\t// Yes, we're doing this 6 times (once per vertex), but on actual GPU hardware,\n\t// it's still faster than doing it in JS combined with the cost of uniformMatrix4fv.\n\t// Expand line bounds by sqrt(2) / 2 each side-- this ensures that all antialiased pixels\n\t// fall within the quad, even at a 45-degree diagonal\n\tvec2 position = a_position;\n\tfloat expandedRadius = (a_lineThicknessAndLength.x * 0.5) + 1.4142135623730951;\n\t// The X coordinate increases along the length of the line. It's 0 at the center of the origin point\n\t// and is in pixel-space (so at n pixels along the line, its value is n).\n\tv_texCoord.x = mix(0.0, a_lineThicknessAndLength.y + (expandedRadius * 2.0), a_position.x) - expandedRadius;\n\t// The Y coordinate is perpendicular to the line. It's also in pixel-space.\n\tv_texCoord.y = ((a_position.y - 0.5) * expandedRadius) + 0.5;\n\tposition.x *= a_lineThicknessAndLength.y + (2.0 * expandedRadius);\n\tposition.y *= 2.0 * expandedRadius;\n\t// 1. Center around first pen point\n\tposition -= expandedRadius;\n\t// 2. Rotate quad to line angle\n\tvec2 pointDiff = a_penPoints.zw;\n\t// Ensure line has a nonzero length so it's rendered properly\n\t// As long as either component is nonzero, the line length will be nonzero\n\t// If the line is zero-length, give it a bit of horizontal length\n\tpointDiff.x = (abs(pointDiff.x) < epsilon && abs(pointDiff.y) < epsilon) ? epsilon : pointDiff.x;\n\t// The `normalized` vector holds rotational values equivalent to sine/cosine\n\t// We're applying the standard rotation matrix formula to the position to rotate the quad to the line angle\n\t// pointDiff can hold large values so we must divide by u_lineLength instead of calling GLSL's normalize function:\n\t// https://asawicki.info/news_1596_watch_out_for_reduced_precision_normalizelength_in_opengl_es\n\tvec2 normalized = pointDiff / max(a_lineThicknessAndLength.y, epsilon);\n\tposition = mat2(normalized.x, normalized.y, -normalized.y, normalized.x) * position;\n\t// 3. Translate quad\n\tposition += a_penPoints.xy;\n\t// 4. Apply view transform\n\tposition *= 2.0 / u_stageSize;\n\tgl_Position = vec4(position, 0.0, 1.0);\n\tv_lineColor = a_lineColor;\n\tv_lineThickness = a_lineThicknessAndLength.x;\n\tv_lineLength = a_lineThicknessAndLength.y;\n\tv_penPoints = a_penPoints;\n\t#elif defined(DRAW_MODE_background)\n\tgl_Position = vec4(a_position * 2.0, 0.0, 1.0);\n\t#else\n\tgl_Position = u_projectionMatrix * u_modelMatrix * vec4(a_position, 0.0, 1.0);\n\tv_texCoord = a_texCoord;\n\t#endif\n}";
    if (!Scratch.extensions.unsandboxed) {
        throw new Error("Colorify must be run unsandboxed");
    }
    const vm = Scratch.vm;
    vm.exports.RenderedTarget.prototype.tintR = 255;
    vm.exports.RenderedTarget.prototype.tintG = 255;
    vm.exports.RenderedTarget.prototype.tintB = 255;
    vm.exports.RenderedTarget.prototype.tintA = 255;
    vm.exports.RenderedTarget.prototype.tileX = 1;
    vm.exports.RenderedTarget.prototype.tileY = 1;
    vm.exports.RenderedTarget.prototype.blendMode = 0;
    vm.exports.RenderedTarget.prototype.maskingMode = 0;
    const runtime = vm.runtime;
    const twgl = runtime.renderer.exports.twgl;
    const renderer = runtime.renderer;
    const gl = renderer._gl;
    gl.enable(gl.STENCIL_TEST);
    const PenSkin = renderer.exports.PenSkin.prototype;
    const resizeables = [];
    const resizeablesFunction = [];
    const nullShaderFsVs = [
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nvoid main()\
        \n{\
        \n\t    gl_FragColor = texture2D(u_texture, v_texcoord);\
        \n}"
    ];
    const nullShader = twgl.createProgramInfo(gl, nullShaderFsVs);

    const nullDiscardShader = twgl.createProgramInfo(gl, ["precision mediump float;\n"+
    "attribute vec4 a_position;\n"+
    "attribute vec2 a_texcoord;\n"+
    "varying vec2 v_texcoord;\n"+
    "void main() {\n\t"+
    "gl_Position = a_position;\n\t"+
    "v_texcoord = a_texcoord;\n\t"+
    "}",


    
    "precision mediump float;\
    \nvarying vec2 v_texcoord;\
    \nuniform sampler2D u_texture;\
    \nvoid main()\
    \n{\
    \n\t    vec4 texture = texture2D(u_texture, v_texcoord);\
    \n\t    if(texture.a == 0.0){\
    \n\t\t      discard;\
    \n\t    }\
    \n\t    gl_FragColor = texture;\
    \n}"]);

    //https://github.com/jamieowen/glsl-blend
    const OverlayBlend = twgl.createProgramInfo(gl, [
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform sampler2D u_texture1;\
        \nfloat blendOverlay(float base, float blend) {\
        \n    return base<0.5?(2.0*base*blend):(1.0-2.0*(1.0-base)*(1.0-blend));\
        \n}\
        \nvec3 blendOverlay(vec3 base, vec3 blend) {\
        \n    return vec3(blendOverlay(base.r,blend.r),blendOverlay(base.g,blend.g),blendOverlay(base.b,blend.b));\
        \n}\
        \nvec3 blendOverlay(vec3 base, vec3 blend, float opacity) {\
        \n    return (blendOverlay(blend,base) * opacity + base * (1.0 - opacity));\
        \n}\
        \nvoid main()\
        \n{\
        \n\t    vec4 base = texture2D(u_texture1, v_texcoord);\
        \n\t    vec4 blend = texture2D(u_texture, v_texcoord);\
        \n\t    if (base.a == 0.0 && blend.a == 0.0){\
        \n\t\t        discard;\
        \n\t    }\
        \n\t    gl_FragColor = vec4(blendOverlay(base.xyz, blend.xyz, blend.w), 1.0);\
        \n}"
    ]);
    const SoftLightBlend = twgl.createProgramInfo(gl, [
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform sampler2D u_texture1;\
        \nfloat blendSoftLight(float base, float blend) {\
        \n\t    return (blend<0.5)?(2.0*base*blend+base*base*(1.0-2.0*blend)):(sqrt(base)*(2.0*blend-1.0)+2.0*base*(1.0-blend));\
        \n}\
        \nvec3 blendSoftLight(vec3 base, vec3 blend) {\
        \n\t    return vec3(blendSoftLight(base.r,blend.r),blendSoftLight(base.g,blend.g),blendSoftLight(base.b,blend.b));\
        \n}\
        \nvec3 blendSoftLight(vec3 base, vec3 blend, float opacity) {\
        \n\t    return (blendSoftLight(blend,base) * opacity + base * (1.0 - opacity));\
        \n}\
        \nvoid main()\
        \n{\
        \n\t    vec4 base = texture2D(u_texture1, v_texcoord);\
        \n\t    vec4 blend = texture2D(u_texture, v_texcoord);\
        \n\t    if (base.a == 0.0 && blend.a == 0.0){\
        \n\t\t        discard;\
        \n\t    }\
        \n\t    gl_FragColor = vec4(blendSoftLight(base.xyz, blend.xyz, blend.w), 1.0);\
        \n}"
    ]);
    const DifferenceBlend = twgl.createProgramInfo(gl, [
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform sampler2D u_texture1;\
        \nvec3 blendDifference(vec3 base, vec3 blend) {\
        \n\t    return abs(base-blend);\
        \n}\
        \nvec3 blendDifference(vec3 base, vec3 blend, float opacity) {\
        \n\t    return (blendDifference(base, blend) * opacity + base * (1.0 - opacity));\
        \n}\
        \nvoid main()\
        \n{\
        \n\t    vec4 base = texture2D(u_texture1, v_texcoord);\
        \n\t    vec4 blend = texture2D(u_texture, v_texcoord);\
        \n\t    if (base.a == 0.0 && blend.a == 0.0){\
        \n\t\t        discard;\
        \n\t    }\
        \n\t    gl_FragColor = vec4(blendDifference(base.xyz, blend.xyz, blend.w), 1.0);\
        \n}"
    ]);
    const MultiplyBlend = twgl.createProgramInfo(gl, [
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform sampler2D u_texture1;\
        \nvec3 blendMultiply(vec3 base, vec3 blend) {\
        \n    return base*blend;\
        \n}\
        \nvec3 blendMultiply(vec3 base, vec3 blend, float opacity) {\
        \n    return (blendMultiply(base, blend) * opacity + base * (1.0 - opacity));\
        \n}\
        \nvoid main()\
        \n{\
        \n\t    vec4 base = texture2D(u_texture1, v_texcoord);\
        \n\t    vec4 blend = texture2D(u_texture, v_texcoord);\
        \n\t    if (base.a == 0.0 && blend.a == 0.0){\
        \n\t\t        discard;\
        \n\t    }\
        \n\t    gl_FragColor = vec4(blendMultiply(base.xyz, blend.xyz, blend.w), 1.0);\
        \n}"
    ]);


    const createPostProcessingFilter = function(shaderVsFs, sizeFunction = null, includeStencilBuffer = false){
        
        var shader = null
        if(shaderVsFs[0] == nullShaderFsVs[0] && shaderVsFs[1] == nullShaderFsVs[1]){
            shader = nullShader;
        }
        else{
            shader = twgl.createProgramInfo(gl, shaderVsFs);
        }
        const tex = twgl.createTexture(gl, {width:gl.canvas.width, height:gl.canvas.height, auto:true});
        var attachments = [{ format: gl.RGBA, type: gl.UNSIGNED_BYTE, min: gl.LINEAR, wrap: gl.CLAMP_TO_EDGE, attachment: tex}];
        const scene = twgl.createFramebufferInfo(gl, attachments, gl.canvas.width, gl.canvas.height);
        twgl.bindFramebufferInfo(gl, scene);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.canvas.width, gl.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
        gl.bindTexture(gl.TEXTURE_2D, null);
        var stencilBuffer = null;
        if (includeStencilBuffer){
            stencilBuffer = gl.createRenderbuffer();
            gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBuffer);
            gl.renderbufferStorage(gl.RENDERBUFFER, gl.STENCIL_INDEX8, gl.canvas.width, gl.canvas.height);
            gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBuffer);
        }
        
        twgl.bindFramebufferInfo(gl, null);
        var result = [scene, shader, tex, stencilBuffer];


        resizeables.push([scene, stencilBuffer]);
        
        if (sizeFunction != null) {
            resizeablesFunction.push(sizeFunction);
        }


        return result;
    }
    const bind = function(scene){
        if (scene != null)
            twgl.bindFramebufferInfo(gl, scene[0]);
        else
            twgl.bindFramebufferInfo(gl, null);
    }
    const drawPostProcessingFilter = function(scene, screen, uniform, drawBuffer, shader = null){
        var drawTargetScene = null;
        var drawTargetShader = null;
        if(scene != null){
            drawTargetScene = scene[0];
            drawTargetShader = scene[1];
        } 
        if (shader != null){
            drawTargetShader = shader;
        }
        if (drawTargetShader == null)
            drawTargetShader = nullShader;
        twgl.bindFramebufferInfo(gl, drawTargetScene);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(1,1,1,1);
        gl.colorMask(true, true, true, true);
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.useProgram(drawTargetShader.program);
        twgl.setBuffersAndAttributes(gl, drawTargetShader, drawBuffer);
        if (uniform == null)
            uniform = {};
        if (screen != null && uniform.u_texture === undefined)
            uniform.u_texture = screen

        
        twgl.setUniforms(drawTargetShader, uniform);
        twgl.drawBufferInfo(gl, drawBuffer, gl.TRIANGLES);
    }
    const drawTexture = function(texture, drawBuffer, shader = nullShader, uniformOpt = null){
        gl.useProgram(shader.program);
        twgl.setBuffersAndAttributes(gl, shader, drawBuffer);
        var uniform = {"u_texture": texture};
        if (uniformOpt != null)
            uniform = uniformOpt;
        
        twgl.setUniforms(shader, uniform);
        twgl.drawBufferInfo(gl, drawBuffer, gl.TRIANGLES);
    }
    const quad = {
        a_position: {
            numComponents: 2,
            data: [
                -1, -1,
                1, -1,
                -1, 1,
                -1, 1,
                1, -1,
                1, 1
            ]
        },
        a_texcoord: {
            numComponents: 2,
            data: [
                0, 0,
                1, 0,
                0, 1,
                
                0, 1,
                1, 0,
                1, 1
              
            ]
        }
        
    };
    const drawBuffer = twgl.createBufferInfoFromArrays(gl, quad);

    const FBOTexSave = createPostProcessingFilter(nullShaderFsVs);

    const FBOsavedStencil = createPostProcessingFilter(nullShaderFsVs);
    

    const saveStencilBufferIntoTexture = function(FBO){
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilMask(0x00);
        gl.colorMask(true, true, true, true);

        const whiteTexture = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, whiteTexture);
        const whitePixel = new Uint8Array([255, 255, 255, 255]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.canvas.width, gl.canvas.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, whitePixel);
        gl.bindTexture(gl.TEXTURE_2D, null);


        twgl.bindFramebufferInfo(gl, FBOTexSave[0]); //save fbo texture
        gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT); 
        drawTexture(FBO.attachments[0], drawBuffer); 

        twgl.bindFramebufferInfo(gl, FBO);//clear fbo (only color, not stencil)
        gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT); 

        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP); //visible in mask
        gl.stencilFunc(gl.EQUAL, 1, 0xff);
        gl.stencilMask(0x00);
        
        drawTexture(whiteTexture, drawBuffer); //draw white box into fbo with stencil mask

        twgl.bindFramebufferInfo(gl, FBOsavedStencil[0]); 
        gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT); 
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        drawTexture(FBO.attachments[0], drawBuffer);     //save fbo's texture(it represents stencil)

        twgl.bindFramebufferInfo(gl, FBO); //restore previous texture of fbo
        gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT); 

        drawTexture(FBOTexSave[0].attachments[0], drawBuffer);
    }

    const restoreStencilBufferFromTexture = function(FBO){
        twgl.bindFramebufferInfo(gl, FBO);//clear fbo (only stencil)
        gl.viewport(0,0,gl.canvas.width,gl.canvas.height);
        gl.clearStencil(0);
        gl.stencilMask(0xFF);
        gl.clear(gl.STENCIL_BUFFER_BIT); 

        gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE); //Mask
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilMask(0xff);
        gl.colorMask(false, false, false, false);

        drawTexture(FBOsavedStencil[0].attachments[0], drawBuffer, nullDiscardShader);


    }



    
    const pingPongFrame = createPostProcessingFilter(nullShaderFsVs);
    
    const pingPongFramePen = createPostProcessingFilter(nullShaderFsVs);

    const pingPongBlendingInternal = createPostProcessingFilter(nullShaderFsVs);
    


    const penTexture = createPostProcessingFilter(nullShaderFsVs, ()=>{
        if (renderer._allSkins[renderer._penSkinId]){
            renderer._allSkins[renderer._penSkinId]._setCanvasSize ([])
        }

        var stencilBufferF = penTexture[3];
        gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBufferF);
        twgl.bindFramebufferInfo(gl, pingPongFramePen[0]); 
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBufferF);

        
    }, true);
    var penTextureI = resizeables.length-1;

    const penTexture_tile = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform float u_tileX;\
        \nuniform float u_tileY;\
        \nvoid main()\
        \n{\
        \n\t    gl_FragColor = texture2D(u_texture, vec2(mod(v_texcoord.x*u_tileX, 1.0), mod(v_texcoord.y*u_tileY, 1.0)));\
        \n}"
    ]);
    const gaussianBlur = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform float u_distortion;\
        \nuniform vec2 u_textureSize;\
        \nvoid main()\
        \n{\
        \n\t    vec3 final_color = vec3(0.0);\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-2.0 * u_distortion, -2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-2.0 * u_distortion, -u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-2.0 * u_distortion, 0.0)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-2.0 * u_distortion, u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-2.0 * u_distortion, 2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-u_distortion, -2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-u_distortion, -u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-u_distortion, 0.0)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-u_distortion, u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(-u_distortion, 2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(0.0, -2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(0.0, -u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(0.0, 0.0)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(0.0, u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(0.0, 2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(u_distortion, -2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(u_distortion, -u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(u_distortion, 0.0)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(u_distortion, u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(u_distortion, 2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(2.0 * u_distortion, -2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(2.0 * u_distortion, -u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(2.0 * u_distortion, 0.0)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(2.0 * u_distortion, u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    final_color += texture2D(u_texture, v_texcoord+vec2(2.0 * u_distortion, 2.0 * u_distortion)/u_textureSize).rgb  * 0.16;\
        \n\t    gl_FragColor = vec4(final_color/4.0, 1.0);\
        \n}"
    ]);
    const blurryBlur = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform float u_distortion;\
        \nuniform vec2 u_textureSize;\
        \nvoid main()\
        \n{\
        \n\t    float stepU = (1.0 / u_textureSize.x) * u_distortion;\
        \n\t    float stepV = (1.0 / u_textureSize.y) * u_distortion;\
        \n\t    vec3 result = vec3(0.0);\
        \n\t    result += 1.0 * texture2D(u_texture, v_texcoord+vec2(-stepU, -stepV)).rgb;\
        \n\t    result += 2.0 * texture2D(u_texture, v_texcoord+vec2(-stepU, 0.0)).rgb;\
        \n\t    result += 1.0 * texture2D(u_texture, v_texcoord+vec2(-stepU, stepV)).rgb;\
        \n\t    result += 2.0 * texture2D(u_texture, v_texcoord+vec2(0.0, -stepV)).rgb;\
        \n\t    result += 4.0 * texture2D(u_texture, v_texcoord+vec2(0.0, 0.0)).rgb;\
        \n\t    result += 2.0 * texture2D(u_texture, v_texcoord+vec2(0.0, stepV)).rgb;\
        \n\t    result += 1.0 * texture2D(u_texture, v_texcoord+vec2(stepU, -stepV)).rgb;\
        \n\t    result += 2.0 * texture2D(u_texture, v_texcoord+vec2(stepU, 0.0)).rgb;\
        \n\t    result += 1.0 * texture2D(u_texture, v_texcoord+vec2(stepU, stepV)).rgb;\
        \n\t    gl_FragColor = vec4(result/16.0, 1.0);\
        \n}"
    ]);
    // const blurFocus = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nuniform vec2 u_textureSize;\
    //     \nuniform float u_size;\
    //     \nconst float MAX_ITERATIONS = 100.0;\
    //     \nvoid main()\
    //     \n{\
    //     \n\t    vec3 result = vec3(0.0);\
    //     \n\t    for(float u = 0.0 ; u < MAX_ITERATIONS; u += 0.2) {\
    //     \n\t\t        if (!(u < u_size)) {break;}\
    //     \n\t\t        vec2 center = v_texcoord - vec2(0.5, 0.5);\
    //     \n\t\t        result+=texture2D(u_texture, v_texcoord + center * dot(center, center) * u * 0.5).rgb;\
    //     \n\t    }\
    //     \n\t    gl_FragColor = vec4(result/(u_size*5.0), 1.0);\
    //     \n}"
    // ]);


    // const lightWater = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nuniform float u_time;\
    //     \nfloat col(vec2 coord, float timeX, float intensity)\
    //     \n{\
    //     \n\t    float time = timeX*1.3;\
    //     \n\t    float col = 0.0;\
    //     \n\t    float theta = 0.0;\
    //     \n\t    for (float i = 0.0; i < 8.0; i+=1.0)\
    //     \n\t    {\
    //     \n\t\t        vec2 adjc = coord;\
    //     \n\t\t        theta = 0.8975979*i;\
    //     \n\t\t        adjc.x += cos(theta)*time*0.2 + time * 0.2;\
    //     \n\t\t        adjc.y -= sin(theta)*time*0.2 - time * 0.3;\
    //     \n\t\t        col = col + cos( (adjc.x*cos(theta) - adjc.y*sin(theta))*6.0)*intensity;\
    //     \n\t    }\
    //     \n\t    return cos(col);\
    //     \n}\
    //     \nvoid main()\
    //     \n{\
    //     \n\t    vec2 p = v_texcoord;\
    //     \n\t    vec2 c1 = p;\
    //     \n\t    vec2 c2 = p;\
    //     \n\t    float cc1 = col(c1, u_time, u_distortion);\
    //     \n\t    c2.x += 8.53;\
    //     \n\t    float dx =  0.5*(cc1-col(c2, u_time, u_distortion))/60.0;\
    //     \n\t    c2.x = p.x;\
    //     \n\t    c2.y += 8.53;\
    //     \n\t    float dy =  0.5*(cc1-col(c2, u_time, u_distortion))/60.0;\
    //     \n\t    c1.x += dx*2.0;\
    //     \n\t    c1.y = (c1.y+dy*2.0);\
    //     \n\t    float alpha = 1.0+dot(dx,dy)*700.0;\
    //     \n\t    float ddx = dx - 0.012;\
    //     \n\t    float ddy = dy - 0.012;\
    //     \n\t    if (ddx > 0.0 && ddy > 0.0) alpha = pow(alpha, ddx*ddy*200000.0);\
    //     \n\t    vec4 col = texture2D(u_texture,c1)*(alpha);\
    //     \n\t    gl_FragColor = col;\
    //     \n}"
    // ]);

    // const waves = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nuniform float u_time;\
    //     \nvoid main()\
    //     \n{\
    //     \n\t    vec2 texcoord = v_texcoord;\
    //     \n\t    texcoord.x += (sin((texcoord.y + (u_time * 0.07)) * u_distortion) * 0.009) + (sin((texcoord.y + (u_time * 0.1)) * (u_distortion-10.0)) * 0.005);\
    //     \n\t    gl_FragColor = texture2D(u_texture, texcoord);\
    //     \n}"
    // ]);


    const blurHoriz = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform vec2 u_textureSize;\
        \nuniform float u_amount;\
        \nuniform float u_glow;\
        \nvoid main()\
        \n{\
        \n\t    float horizontalTexel = 1.0 / u_textureSize.x * u_amount;\
        \n\t    vec4 result = texture2D(u_texture, v_texcoord) * 0.204163688 * u_glow;\
        \
        \n\t    vec2 offset = vec2(horizontalTexel, 0.0); \
        \n\t    vec2 samleCoordsRight = v_texcoord + offset;\
        \n\t    vec2 samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.180173822 * u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.180173822 * u_glow;\
        \
        \n\t    offset = vec2(horizontalTexel*2.0, 0.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.123831536 * u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.123831536 * u_glow;\
        \
        \n\t    offset = vec2(horizontalTexel*3.0, 0.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.066282245 * u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.066282245 * u_glow;\
        \
        \n\t    offset = vec2(horizontalTexel*4.0, 0.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.027630550 * u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.027630550 * u_glow;\
        \
        \n\t    gl_FragColor = vec4(result.rgb, 1.0);\
        \n}"
    ]);
    const blurVertic = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform vec2 u_textureSize;\
        \nuniform float u_amount;\
        \nuniform float u_glow;\
        \nvoid main()\
        \n{\
        \n\t    float horizontalTexel = 1.0 / u_textureSize.y * u_amount;\
        \n\t    vec4 result = texture2D(u_texture, v_texcoord) * 0.204163688 * u_glow;\
        \
        \n\t    vec2 offset = vec2(0.0, horizontalTexel); \
        \n\t    vec2 samleCoordsRight = v_texcoord + offset;\
        \n\t    vec2 samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.180173822* u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.180173822* u_glow;\
        \
        \n\t    offset = vec2(0.0, horizontalTexel*2.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.123831536* u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.123831536* u_glow;\
        \
        \n\t    offset = vec2(0.0, horizontalTexel*3.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.066282245* u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.066282245* u_glow;\
        \
        \n\t    offset = vec2(0.0, horizontalTexel*4.0); \
        \n\t    samleCoordsRight = v_texcoord + offset;\
        \n\t    samleCoordsLeft = v_texcoord - offset;\
        \n\t    result += texture2D(u_texture, samleCoordsRight) * 0.027630550* u_glow;   \
        \n\t    result += texture2D(u_texture, samleCoordsLeft) * 0.027630550* u_glow;\
        \
        \n\t    gl_FragColor = vec4(result.rgb, 1.0);\
        \n}"
    ]);
    const bloomBright = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform vec3 u_color;\
        \nuniform sampler2D u_texture;\
        \nvec3 brightness;\
        \nuniform float u_brightnessThreshold;\
        \nuniform float u_intensity;\
        \nvoid main()\
        \n{\
        \n\t    vec4 fragColor = texture2D(u_texture, v_texcoord) * vec4(u_color, 1.0);\
        \n\t    float br = dot(fragColor.rgb, vec3(0.299, 0.587, 0.114));\
        \n\t    if(br > u_brightnessThreshold)\
        \n\t    {\
        \n\t\t        brightness = fragColor.rgb;\
        \n\t    }\
        \n\t    else \
        \n\t    {\
        \n\t\t        brightness = vec3(0.0);\
        \n\t    }\
        \n\t    gl_FragColor = vec4(brightness*u_intensity, 1.0);\
        \n}"
    ]);
    const bloom = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture;\
        \nuniform sampler2D u_texture1;\
        \nvoid main() {\
        \n\t    vec4 t0 = texture2D(u_texture, v_texcoord); // screen\
        \n\t    vec4 t1 = texture2D(u_texture1, v_texcoord); // bloom\
        \n\t    t0 *= (vec4(1.0) - t1);\
        \n\t    gl_FragColor = t0 + t1;\
        \n}"
    ]);
    
    const grayscale = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        "precision mediump float;\
        \nuniform float desaturationFactor;\
        \nuniform sampler2D u_texture;\
        \nvarying vec2 v_texcoord;\
        \nvoid main()\
        \n{\
        \n\t  vec4 color = texture2D( u_texture , v_texcoord );\
        \n\t  vec3 gray = vec3( dot( color.rgb , vec3( 0.2126 , 0.7152 , 0.0722 ) ) );\
        \n\t  gl_FragColor = vec4(mix(color.rgb,gray, desaturationFactor) , color.a );\
        \n}"
    ], ()=>{                                                   
        var stencilBufferF = grayscale[3];
        gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBufferF);
        twgl.bindFramebufferInfo(gl, pingPongFrame[0]); 
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBufferF);
    }, true);
    var grayscaleI = resizeables.length-1;
    
    var stencilBuffeR = grayscale[3];
    gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBuffeR);
    twgl.bindFramebufferInfo(gl, pingPongFrame[0]); 
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBuffeR);
    stencilBuffeR = penTexture[3];
    gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBuffeR);
    twgl.bindFramebufferInfo(gl, pingPongFramePen[0]); 
    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBuffeR);


    const aberration = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        "precision mediump float;\
        \nuniform float degree;\
        \nuniform sampler2D u_texture;\
        \nvarying vec2 v_texcoord;\
        \nvoid main()\
        \n{\
        \n\t  float r = texture2D(u_texture, v_texcoord+vec2(degree, 0)).r;\
        \n\t  float g = texture2D(u_texture, v_texcoord).g;\
        \n\t  float b = texture2D(u_texture, v_texcoord+vec2(0, degree)).b;\
        \n\t  gl_FragColor = vec4(r, g, b, 1.0);\
        \n}"
    ]);


    const fisheye = createPostProcessingFilter([
        "precision mediump float;\n"+
        "attribute vec4 a_position;\n"+
        "attribute vec2 a_texcoord;\n"+
        "varying vec2 v_texcoord;\n"+
        "void main() {\n\t"+
        "gl_Position = a_position;\n\t"+
        "v_texcoord = a_texcoord;\n\t"+
        "}",


        
        "precision mediump float;\
        \nvarying vec2 v_texcoord;\
        \nuniform sampler2D u_texture; // screen texture\
        \nuniform vec2 u_textureSize;\
        \nuniform float u_distortion;\
        \nvoid main() \
        \n{\
        \n\t    vec2 p = v_texcoord.xy / u_textureSize.x;\
        \n\t    float prop = u_textureSize.x / u_textureSize.y;\
        \n\t    vec2 m = vec2(0.5, 0.5 / prop);\
        \n\t    vec2 d = p - m;\
        \n\t    float r = sqrt(dot(d, d));\
        \n\t    float power = 4.44289334 * (u_distortion - 0.5);\
        \n\t    float bind;\
        \n\t    if (power > 0.0) bind = sqrt(dot(m, m));\
        \n\t    else {if (prop < 1.0) bind = m.x; else bind = m.y;}\
        \n\t    vec2 uv;\
        \n\t    if (power > 0.0)\
        \n\t\t        uv = m + normalize(d) * tan(r * power) * bind / tan( bind * power);\
        \n\t    else if (power < 0.0)\
        \n\t\t        uv = m + normalize(d) * atan(r * -power * 10.0) * bind / atan(-power * bind * 10.0);\
        \n\t    else uv = p;\
        \n\t    gl_FragColor = texture2D(u_texture, vec2(uv.x, uv.y * prop));\
        \n}"
    ]);





    // const Contrast = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nvoid main() \
    //     \n{\
    //     \n\t    vec3 tex = texture2D(u_texture, v_texcoord).rgb;\
    //     \n\t    float value = dot(tex,vec3(0.2126,0.7152,0.0722));\
    //     \n\t    gl_FragColor = vec4(mix(vec3(0.5,0.5,0.5), tex, u_distortion), 1.0);\
    //     \n}"
    // ]);

    // const Sepia = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nvoid main() \
    //     \n{\
    //     \n\t    vec3 tex = texture2D(u_texture, v_texcoord).rgb;\
    //     \n\t    vec3 d = vec3(dot(tex,vec3(0.222,0.707,0.071)));\
    //     \n\t    d.r+=0.437;\
    //     \n\t    d.g+=0.171;\
    //     \n\t    d.b+=0.078;\
    //     \n\t    gl_FragColor = vec4(mix(tex, d, u_distortion), 1.0);\
    //     \n}"
    // ]);
    // const static = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nuniform float u_time;\
    //     \nvoid main()\
    //     \n{\
    //     \n\t    gl_FragColor = mix(texture2D(u_texture, v_texcoord), texture2D(u_texture, vec2(v_texcoord.x+u_time*10.0, v_texcoord.y+u_time*0.5)), u_distortion);\
    //     \n}"
    // ]);

    // const sharpen = createPostProcessingFilter([
    //     "precision mediump float;\n"+
    //     "attribute vec4 a_position;\n"+
    //     "attribute vec2 a_texcoord;\n"+
    //     "varying vec2 v_texcoord;\n"+
    //     "void main() {\n\t"+
    //     "gl_Position = a_position;\n\t"+
    //     "v_texcoord = a_texcoord;\n\t"+
    //     "}",


        
    //     "precision mediump float;\
    //     \nvarying vec2 v_texcoord;\
    //     \nuniform vec2 u_textureSize;\
    //     \nuniform sampler2D u_texture;\
    //     \nuniform float u_distortion;\
    //     \nvoid main()\
    //     \n{\
    //     \n\t    float r = u_distortion/u_textureSize.x; \
    //     \n\t    vec2 vert = vec2(r,0.0);\
    //     \n\t    vec2 horiz = vec2(0.0, r);\
    //     \n\t    vec4 c0 = texture2D(u_texture,v_texcoord);\
    //     \n\t    vec4 c1 = texture2D(u_texture,v_texcoord-vert);\
    //     \n\t    vec4 c2 = texture2D(u_texture,v_texcoord+vert);\
    //     \n\t    vec4 c3 = texture2D(u_texture,v_texcoord-horiz);\
    //     \n\t    vec4 c4 = texture2D(u_texture,v_texcoord+horiz);\
    //     \n\t    vec4 c5 = (c0+c1+c2+c3+c4)*0.2;\
    //     \n\t    vec4 mi = min(c0,c1);\
    //     \n\t    mi = min(mi,c2);\
    //     \n\t    mi = min(mi,c3);\
    //     \n\t    mi = min(mi,c4);\
    //     \n\t    vec4 ma = max(c0,c1);\
    //     \n\t    ma = max(ma,c2);\
    //     \n\t    ma = max(ma,c3);\
    //     \n\t    ma = max(ma,c4);\
    //     \n\t    gl_FragColor = clamp(mi,37.0*c0-36.0*c5,ma);\
    //     \n}"
    // ]);



    
    

    var pro = Object.getPrototypeOf(renderer._shaderManager);
    pro.constructor.EFFECT_INFO.u_tintR = {
        uniformName: 'u_tintR',
        mask: 1 << 7,
        converter: x => 0,
        shapeChanges: false
    } 
    pro.constructor.EFFECT_INFO.u_tintG = {
        uniformName: 'u_tintG',
        mask: 1 << 7,
        converter: x => 0,
        shapeChanges: false
    } 
    pro.constructor.EFFECT_INFO.u_tintB = {
        uniformName: 'u_tintB',
        mask: 1 << 7,
        converter: x => 0,
        shapeChanges: false
    }
    pro.constructor.EFFECT_INFO.u_tintA = {
        uniformName: 'u_tintA',
        mask: 1 << 7,
        converter: x => 0,
        shapeChanges: false
    }
    pro.constructor.EFFECT_INFO.u_tileX = {
        uniformName: 'u_tileX',
        mask: 1 << 8,
        converter: x => 1,
        shapeChanges: true
    }
    pro.constructor.EFFECT_INFO.u_tileY = {
        uniformName: 'u_tileY',
        mask: 1 << 8,
        converter: x => 1,
        shapeChanges: true
    }
    pro.constructor.EFFECT_INFO.u_blendMode = {
        uniformName: 'u_blendMode',
        mask: 1 << 9,
        converter: x => 0,
        shapeChanges: true
    }
    pro.constructor.EFFECT_INFO.u_maskingMode = {
        uniformName: 'u_maskingMode',
        mask: 1 << 9,
        converter: x => 0,
        shapeChanges: true
    }
    pro.constructor.EFFECTS.push("u_tintR");
    pro.constructor.EFFECTS.push("u_tintG"); 
    pro.constructor.EFFECTS.push("u_tintB"); 
    pro.constructor.EFFECTS.push("u_tintA"); 
    pro.constructor.EFFECTS.push("u_tileX"); 
    pro.constructor.EFFECTS.push("u_tileY"); 
    pro.constructor.EFFECTS.push("u_blendMode"); 
    pro.constructor.EFFECTS.push("u_maskingMode"); 


    renderer._allDrawables.forEach(element => {
        element._uniforms.u_tintR = 0;
        element._uniforms.u_tintG = 0;
        element._uniforms.u_tintB = 0;
        element._uniforms.u_tintA = 0;
        element._uniforms.u_tileX = 1;
        element._uniforms.u_tileY = 1;
        element._uniforms.u_blendMode = 0;
        element._uniforms.u_maskingMode = 0;
    });
    pro._buildShader = function(drawMode, effectBits) {
        const numEffects = 7;

        const defines = [
            `#define DRAW_MODE_${drawMode}`
        ];
        for (let index = 0; index < numEffects; ++index) {
            if ((effectBits & (1 << index)) !== 0) {
                defines.push(`#define ENABLE_${pro.constructor.EFFECTS[index]}`);
            }
        }

        const definesText = `${defines.join('\n')}\n`;

        /* eslint-disable global-require */
        const vsFullText = definesText + vsShader;
        const fsFullText = definesText + fragShader;
        /* eslint-enable global-require */

        return twgl.createProgramInfo(gl, [vsFullText, fsFullText]);
    }
    renderer._shaderManager._shaderCache = {};
    var DRAW_MODE = {
        default: 'default',
        straightAlpha: 'straightAlpha',
        silhouette: 'silhouette',
        colorMask: 'colorMask',
        line: 'line',
        background: 'background'
    };
    for (const modeName in DRAW_MODE) {
        renderer._shaderManager._shaderCache[modeName] = [];
    }

    
    var ren = Object.getPrototypeOf(renderer);
    var lastWidth = gl.canvas.width;
    var lastHeight = gl.canvas.height;
    var u_brightnessThreshold = 1;
    var u_intensity = 1;
    var u_color = [1,1,1];
    var aberration_degree = 0;
    var desaturationFactor2 = 0;
    var u_fisheye = 0.5;
    var screenTileX = 1;
    var screenTileY = 1;
    var u_lightWater = 0.0;
    var u_time = 0.0;

    function loadImage (url, callback) {
        var image = new Image();
        image.crossOrigin = "";
        image.onload = callback;
        image.src = url;
        return image;
      }
    function loadImages(urls, callback) {
        var images = [];
        var imagesToLoad = urls.length;
        var onImageLoad = function() {
          --imagesToLoad;
          if (imagesToLoad === 0) {
            callback(images);
          }
        };
       
        for (var ii = 0; ii < imagesToLoad; ++ii) {
          var image = loadImage(urls[ii], onImageLoad);
          images.push(image);
        }
      }
    
      
    ren.penClear = function(penSkinID){
        this.dirty = true;
        // const skin = /** @type {PenSkin} */ this._allSkins[penSkinID];
        // skin.clear();
        const gl = this._gl;
        twgl.bindFramebufferInfo(gl, penTexture[0]); 
        gl.clearColor(0, 0, 0, 0);
        gl.clearStencil(0);
        gl.stencilMask(0xff);
        gl.colorMask(true, true, true, true);
        gl.clear(gl.STENCIL_BUFFER_BIT | gl.COLOR_BUFFER_BIT);
        
    };

    
    ren.penStamp = function(penSkinID, stampID) {
        this.dirty = true;
        const stampDrawable = this._allDrawables[stampID];
        if (!stampDrawable) {
            return;
        }
        const skin = /** @type {PenSkin} */ this._allSkins[penSkinID];
        const gl = this._gl;
        this._doExitDrawRegion();
        twgl.bindFramebufferInfo(gl, penTexture[0]); 


        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        this._drawThese([stampID], 'default', this._projection, {
            ignoreVisibility: true
        }, 2);
        skin._silhouetteDirty = true;

    };
    PenSkin._enterDrawLineOnBuffer = function(){
        // tw: reset attributes when starting pen drawing
        this._resetAttributeIndexes();
        const gl = this._renderer.gl;

        twgl.bindFramebufferInfo(gl, this._framebuffer);

        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);

        const currentShader = this._lineShader;
        gl.useProgram(currentShader.program);
        twgl.setBuffersAndAttributes(gl, currentShader, this._lineBufferInfo);
        var stageSize = this._renderer.useHighQualityRender ? [gl.canvas.width, gl.canvas.height] : this._nativeSize;
        const uniforms = {
            u_skin: this._texture,
            u_stageSize: stageSize
        };

        twgl.setUniforms(currentShader, uniforms);
    }
    PenSkin.onNativeSizeChanged = function(event){
        this._setCanvasSize([]);
        this.emitWasAltered();
    }
    PenSkin._setCanvasSize = function (canvasSize) {
        const gl = this._renderer.gl;
        canvasSize = [gl.canvas.width, gl.canvas.height];

        const [width, height] = canvasSize;

        // tw: do not resize if new size === old size
        if (this._size && this._size[0] === width && this._size[1] === height) {
            return;
        }

        this._size = canvasSize;
        this._nativeSize = renderer.getNativeSize();
        // tw: use native size for Drawable positioning logic
        this._rotationCenter[0] = this._nativeSize[0] / 2;
        this._rotationCenter[1] = this._nativeSize[1] / 2;


        // tw: store current texture to redraw it later
        const oldTexture = this._texture;

        this._texture = penTexture[0].attachments[0];
        
        this._framebuffer = penTexture[0];

        gl.clearColor(0, 0, 0, 0);
        gl.clearStencil(0);
        gl.stencilMask(0xff);
        gl.colorMask(true, true, true, true);
        gl.clear(gl.STENCIL_BUFFER_BIT | gl.COLOR_BUFFER_BIT);

        // tw: preserve old texture when resizing
        if (oldTexture) {
            this._drawPenTexture(oldTexture);
        }

        this._silhouettePixels = new Uint8Array(Math.floor(width * height * 4));
        this._silhouetteImageData = new ImageData(width, height);

        this._silhouetteDirty = true;
        this._renderer.dirty = true
    }
    renderer.createPenSkin();
    ren.penPoint = function(penSkinID, penAttributes, x, y) {
        this.dirty = true;
        const skin = /** @type {PenSkin} */ this._allSkins[penSkinID];
        skin.drawLine(penAttributes, x, -y, x, -y);

    }
    ren.penLine = function(penSkinID, penAttributes, x0, y0, x1, y1) {
        this.dirty = true;
        const skin = /** @type {PenSkin} */ this._allSkins[penSkinID];
        skin.drawLine(penAttributes, x0, -y0, x1, -y1);
    }
    ren.updatePenScreen = function() {
        u_time = (Date.now()/1000)%100;
        const gl = this._gl;
        gl.colorMask(true, true, true, true);
        twgl.bindFramebufferInfo(gl, fisheye[0]);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        drawTexture(null, drawBuffer, fisheye[1], {"u_texture": penTexture[0].attachments[0], "u_distortion": u_fisheye, "u_textureSize":[1, 1]});

        twgl.bindFramebufferInfo(gl, penTexture_tile[0]);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        drawTexture(null, drawBuffer, penTexture_tile[1], {"u_texture": fisheye[0].attachments[0], "u_tileX": screenTileX, "u_tileY": screenTileY});
        
        twgl.bindFramebufferInfo(gl, penTexture[0]);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0,0,0,0);
        gl.clear(gl.COLOR_BUFFER_BIT);
        drawTexture(penTexture_tile[0].attachments[0], drawBuffer);
    }
    ren._drawThese = function(drawables, drawMode, projection, opts = {}, binded = 0) {
        const gl = this._gl;
        
        gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
        gl.stencilFunc(gl.ALWAYS, 1, 0xff);
        gl.stencilMask(0x00);
        gl.colorMask(true, true, true, true);

        gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
        gl.blendEquation(gl.FUNC_ADD)
        

        let currentShader = null;

        const framebufferSpaceScaleDiffers = (
            'framebufferWidth' in opts && 'framebufferHeight' in opts &&
            opts.framebufferWidth !== this._nativeSize[0] && opts.framebufferHeight !== this._nativeSize[1]
        );

        const numDrawables = drawables.length;
        for (let drawableIndex = 0; drawableIndex < numDrawables; ++drawableIndex) {
            const drawableID = drawables[drawableIndex];

            // If we have a filter, check whether the ID fails
            if (opts.filter && !opts.filter(drawableID)) continue;

            const drawable = this._allDrawables[drawableID];
            /** @todo check if drawable is inside the viewport before anything else */

            // Hidden drawables (e.g., by a "hide" block) are not drawn unless
            // the ignoreVisibility flag is used (e.g. for stamping or touchingColor).
            if (!drawable.getVisible() && !opts.ignoreVisibility) continue;

            // drawableScale is the "framebuffer-pixel-space" scale of the drawable, as percentages of the drawable's
            // "native size" (so 100 = same as skin's "native size", 200 = twice "native size").
            // If the framebuffer dimensions are the same as the stage's "native" size, there's no need to calculate it.
            const drawableScale = framebufferSpaceScaleDiffers ? [
                drawable.scale[0] * opts.framebufferWidth / this._nativeSize[0],
                drawable.scale[1] * opts.framebufferHeight / this._nativeSize[1]
            ] : drawable.scale;

            // If the skin or texture isn't ready yet, skip it.
            if (!drawable.skin || !drawable.skin.getTexture(drawableScale)) continue;
            
            if (drawable.skin.id == this._penSkinId){
                drawTexture(penTexture_tile[0].attachments[0], drawBuffer);
                this._regionId = null;
                continue;
            }
            
            // Skip private skins, if requested.
            if (opts.skipPrivateSkins && drawable.skin.private) continue;

            const uniforms = {};
            Object.assign(uniforms,
                drawable.skin.getUniforms(drawableScale),
                drawable.getUniforms());
                
            

            let effectBits = drawable.enabledEffects;

            
            effectBits &= Object.prototype.hasOwnProperty.call(opts, 'effectMask') ? opts.effectMask : effectBits;
            var newShader = null;
            if (uniforms.u_maskingMode == 3.0){
                newShader = this._shaderManager.getShader('silhouette', effectBits);
            }
            else{
                newShader = this._shaderManager.getShader(drawMode, effectBits);
            }

            // Manually perform region check. Do not create functions inside a
            // loop.
            if (this._regionId !== newShader) {
                this._doExitDrawRegion();
                this._regionId = newShader;

                currentShader = newShader;
                gl.useProgram(currentShader.program);
                twgl.setBuffersAndAttributes(gl, currentShader, this._bufferInfo);
                Object.assign(uniforms, {
                    u_projectionMatrix: projection
                });
            }

            

            // Apply extra uniforms after the Drawable's, to allow overwriting.
            if (opts.extraUniforms) {
                Object.assign(uniforms, opts.extraUniforms);
            }

            if (uniforms.u_skin) {
                twgl.setTextureParameters(
                    gl, uniforms.u_skin, {
                        minMag: drawable.skin.useNearest(drawableScale, drawable) ? gl.NEAREST : gl.LINEAR
                    }
                );
            }
            var pingPongRequired = false;
            if(uniforms.u_blendMode == 1.0){
                gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_COLOR, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            }
            else if (uniforms.u_blendMode == 2.0){
                gl.blendFuncSeparate(gl.ONE, gl.ONE, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            }
            else if (uniforms.u_blendMode != 0.0){
                pingPongRequired = true;
            }
            
            if(pingPongRequired){
                if (binded == 1){
                    twgl.bindFramebufferInfo(gl, pingPongFrame[0]);
                    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
                    gl.clearColor(0,0,0,0);
                    gl.colorMask(true, true, true, true);
                    gl.clear(gl.COLOR_BUFFER_BIT);
                }
                else if (binded == 2){
                    twgl.bindFramebufferInfo(gl, pingPongFramePen[0]);
                    gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
                    gl.clearColor(0,0,0,0);
                    gl.colorMask(true, true, true, true);
                    gl.clear(gl.COLOR_BUFFER_BIT);
                }
            }
            

            if (uniforms.u_maskingMode == 3.0){ //Mask
                gl.stencilOp(gl.KEEP, gl.KEEP, gl.REPLACE);
                gl.stencilFunc(gl.ALWAYS, 1, 0xff);
                gl.stencilMask(0xff);
                gl.colorMask(false, false, false, false);
            }
            else if (uniforms.u_maskingMode == 1.0){  //visible in Mask
                gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
                gl.stencilFunc(gl.EQUAL, 1, 0xff);
                gl.stencilMask(0x00);
                gl.colorMask(true, true, true, true);
            }
            else if (uniforms.u_maskingMode == 2.0){  //visible out Mask
                gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
                gl.stencilFunc(gl.NOTEQUAL, 1, 0xff);
                gl.stencilMask(0x00);
                gl.colorMask(true, true, true, true);
            }
            
                
            
            

            twgl.setUniforms(currentShader, uniforms);
            twgl.drawBufferInfo(gl, this._bufferInfo, gl.TRIANGLES);


            gl.stencilOp(gl.KEEP, gl.KEEP, gl.KEEP);
            gl.stencilFunc(gl.ALWAYS, 1, 0xff);
            gl.stencilMask(0x00);
            gl.colorMask(true, true, true, true);

            gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
            gl.blendEquation(gl.FUNC_ADD);


            if(pingPongRequired){
                twgl.bindFramebufferInfo(gl, pingPongBlendingInternal[0]);
                gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
                gl.clearColor(0,0,0,0);
                gl.clear(gl.COLOR_BUFFER_BIT);

                if (binded == 1){
                    switch(uniforms.u_blendMode){
                        case 3.0:
                            drawTexture(null, drawBuffer, OverlayBlend, {u_texture: pingPongFrame[0].attachments[0], u_texture1: grayscale[0].attachments[0]});
                            break;
                        case 4.0:
                            drawTexture(null, drawBuffer, SoftLightBlend, {u_texture: pingPongFrame[0].attachments[0], u_texture1: grayscale[0].attachments[0]});
                            break;
                        case 5.0:
                            drawTexture(null, drawBuffer, DifferenceBlend, {u_texture: pingPongFrame[0].attachments[0], u_texture1: grayscale[0].attachments[0]});
                            break;
                        case 6.0:
                            drawTexture(null, drawBuffer, MultiplyBlend, {u_texture: pingPongFrame[0].attachments[0], u_texture1: grayscale[0].attachments[0]});
                            break;
                    }


                    twgl.bindFramebufferInfo(gl, grayscale[0]);
                }
                else if(binded == 2){
                    switch(uniforms.u_blendMode){
                        case 3.0:
                            drawTexture(null, drawBuffer, OverlayBlend, {u_texture: pingPongFramePen[0].attachments[0], u_texture1: penTexture[0].attachments[0]});
                            break;
                        case 4.0:
                            drawTexture(null, drawBuffer, SoftLightBlend, {u_texture: pingPongFramePen[0].attachments[0], u_texture1: penTexture[0].attachments[0]});
                            break;
                        case 5.0:
                            drawTexture(null, drawBuffer, DifferenceBlend, {u_texture: pingPongFramePen[0].attachments[0], u_texture1: penTexture[0].attachments[0]});
                            break;
                        case 6.0:
                            drawTexture(null, drawBuffer, MultiplyBlend, {u_texture: pingPongFramePen[0].attachments[0], u_texture1: penTexture[0].attachments[0]});
                            break;
                    }


                    twgl.bindFramebufferInfo(gl, penTexture[0]);
                    gl.clearColor(0,0,0,0);
                    gl.clear(gl.COLOR_BUFFER_BIT);
                }

                gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
                drawTexture(pingPongBlendingInternal[0].attachments[0], drawBuffer);
                this._regionId = null;
            }

            

            
            
            

        }

        this._regionId = null;
    }
    var u_blurryBlurDistortion = 0.0;
    var u_focusBlurSize = 0.2;
    var u_distortion = 0.10001; //gaussian blur
    var u_waves = 0;

    // var sepiaSize = 0;
    // var contrastSize = 1;
    // var staticSize = 0;
    // var sharpenSize = 0;



    var autoPenScreenFilter = true;
    ren.draw = function(){
        if (!this.dirty) {
            return;
        }
        this.dirty = false;

        this._doExitDrawRegion();
        
        const gl = this._gl;

        if(lastWidth!=gl.canvas.width||lastHeight!=gl.canvas.height){
            lastWidth = gl.canvas.width;
            lastHeight = gl.canvas.height;
            for (var i = 0; i < resizeables.length; i++){
                gl.bindTexture(gl.TEXTURE_2D, resizeables[i][0].attachments[0]);
                gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, lastWidth, lastHeight, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
                gl.bindTexture(gl.TEXTURE_2D, null);
                if (resizeables[i][1] != null){
                    var stencilBuffer = resizeables[i][1];
                    gl.deleteRenderbuffer(stencilBuffer);
                    stencilBuffer = gl.createRenderbuffer();
                    gl.bindRenderbuffer(gl.RENDERBUFFER, stencilBuffer);
                    gl.renderbufferStorage(gl.RENDERBUFFER, gl.STENCIL_INDEX8, lastWidth, lastHeight);

                    twgl.bindFramebufferInfo(gl, resizeables[i][0]);
                    gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.STENCIL_ATTACHMENT, gl.RENDERBUFFER, stencilBuffer);
                    resizeables[i][1] = stencilBuffer;
                    if (i == grayscaleI){
                        grayscale[3] = stencilBuffer;
                    }
                    else if(i == penTextureI){
                        penTexture[3] = stencilBuffer;
                    }
                    // console.log("resize stencil");
                }
            }
            for (var i = 0; i < resizeablesFunction.length; i++){
                resizeablesFunction[i].call();
            }
            
        }
        if (autoPenScreenFilter) this.updatePenScreen();
        else u_time = Date.now()/1000%100; //ÃÂ¬ÃÂÃÂÃÂ«ÃÂÃÂ updatePenScreenÃÂ¬ÃÂÃÂÃÂ¬ÃÂÃÂ ÃÂ­ÃÂÃÂ´ÃÂ¬ÃÂ¤ÃÂ
        
        twgl.bindFramebufferInfo(gl, grayscale[0]);
        //scratch drawing
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(1,1,1,1);
        gl.clearStencil(0);
        gl.stencilMask(0xff);
        gl.colorMask(true, true, true, true);
        gl.clear(gl.STENCIL_BUFFER_BIT | gl.COLOR_BUFFER_BIT);
        
        const snapshotRequested = this._snapshotCallbacks.length > 0;
        this._drawThese(this._drawList, 'default', this._projection, {
            framebufferWidth: gl.canvas.width,
            framebufferHeight: gl.canvas.height,
            skipPrivateSkins: snapshotRequested
        }, 1); 
        drawPostProcessingFilter(bloom, grayscale[0].attachments[0], {desaturationFactor: desaturationFactor2}, drawBuffer, grayscale[1])
        
        drawPostProcessingFilter(blurryBlur, bloom[0].attachments[0], {u_distortion: u_blurryBlurDistortion, u_textureSize:[gl.canvas.width, gl.canvas.height]}, drawBuffer, blurryBlur[1]);
        
        drawPostProcessingFilter(gaussianBlur, blurryBlur[0].attachments[0], {u_distortion, u_textureSize:[gl.canvas.width, gl.canvas.height]}, drawBuffer, gaussianBlur[1]);

        //post processing bloom tex -> bloomBright shader -> bloomBright
        drawPostProcessingFilter(bloomBright, gaussianBlur[0].attachments[0], {
            u_color,
            u_brightnessThreshold,
            u_intensity
        }, drawBuffer, bloomBright[1])

        drawPostProcessingFilter(blurVertic, bloomBright[0].attachments[0], {u_amount: 1.6, u_glow: 0.9,u_textureSize:[gl.canvas.width, gl.canvas.height]}, drawBuffer, blurHoriz[1]);
        drawPostProcessingFilter(blurHoriz, blurVertic[0].attachments[0], {u_amount: 1.6, u_glow: 0.9,u_textureSize:[gl.canvas.width, gl.canvas.height]}, drawBuffer, blurVertic[1]);
        

        
        drawPostProcessingFilter(aberration, gaussianBlur[0].attachments[0], {u_texture1: blurHoriz[0].attachments[0]}, drawBuffer, bloom[1]);
        
        drawPostProcessingFilter(null, aberration[0].attachments[0], {degree: aberration_degree}, drawBuffer, aberration[1]);




    
        
        if (snapshotRequested) {
            const snapshot = gl.canvas.toDataURL();
            this._snapshotCallbacks.forEach(cb => cb(snapshot));
            this._snapshotCallbacks = [];
            this.dirty = true;
        }
    }
    
    




    
    renderer.dirty = true;
    const BlendParam = {
        None: 'None',
        Screen: 'Screen',
        LinearDodge: 'LinearDodge',
        Overlay: 'Overlay',
        SoftLight: 'SoftLight',
        Difference: 'Difference',
        Multiply: 'Multiply'
    };
    const MaskingParam = {
        None: 'None',
        VisibleInsideMask: 'VisibleInsideMask',
        VisibleOutsideMask: 'VisibleOutsideMask',
        Mask: 'Mask'
    };
    class Colorify {
        constructor () {
            
        }
        getInfo() {
            
            return {
            id: 'colorify1',
            name: 'Colorify',
            blocks: [
                {
                    opcode: 'tint',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'Tint [R][G][B][A]',
                    arguments: {
                        R: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        },
                        G: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        },
                        B: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        },
                        A: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        }
                    }
                },
                {
                    opcode: 'disableAutoPenScreenFilterApplying',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'Disable Auto Pen Screen Filter Applying',
                },
                {
                    opcode: 'updatePenScreenManually',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'Update Pen Screen Manually',
                },
                {
                    opcode: 'savePenStencil',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'Save Pen Stencil Buffer(Mask)',
                },
                {
                    opcode: 'restorePenStencil',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'Restore Pen Stencil Buffer(Mask)',
                },
                {
                    opcode: 'setBlendMode',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setBlendMode [MODE]',
                    arguments: {
                        MODE: {
                            type: Scratch.ArgumentType.STRING,
                            menu: 'blendParam',
                            defaultValue: BlendParam.None
                        }
                    }
                },
                {
                    opcode: 'setMaskingMode',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setMaskingMode [MODE]',
                    arguments: {
                        MODE: {
                            type: Scratch.ArgumentType.STRING,
                            menu: 'maskingParam',
                            defaultValue: MaskingParam.None
                        }
                    }
                },
                {
                    opcode: 'clearMaskInPenScreen',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'clearMaskInPenScreen'
                },
                {
                    opcode: 'setGrayscale',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setGrayscale [value]',
                    arguments: {
                        value: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 100
                        }
                    }
                },
                {
                    opcode: 'setGaussianBlur',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setGaussianBlur [value]',
                    arguments: {
                        value: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 0
                        }
                    }
                },
                
               
                {
                    opcode: 'setBlurryBlur',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setBlurryBlur [value]',
                    arguments: {
                        value: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 0
                        }
                    }
                },
                
                {
                    opcode: 'setAberration',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setAberration [value]',
                    arguments: {
                        value: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 50
                        }
                    }
                },
                {
                    opcode: 'setBloom',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setBloom intensity: [intensity] brightThreshold: [bright] color: ([colorR][colorG][colorB])',
                    arguments: {
                        intensity: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 100
                        },
                        bright: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 100
                        },
                        colorR: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        },
                        colorG: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        },
                        colorB: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 255
                        }
                    }
                },
                {
                    opcode: 'fisheye',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'fisheye [fisheye]',
                    arguments: {
                        fisheye: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 50
                        }
                    }
                },
                {
                    opcode: 'setTiling',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setTiling [X][Y]',
                    arguments: {
                        X: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 1
                        },
                        Y: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 1
                        }
                    }
                },
                {
                    opcode: 'setScreenTiling',
                    blockType: Scratch.BlockType.COMMAND,
                    text: 'setScreenTiling [X][Y]',
                    arguments: {
                        X: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 1
                        },
                        Y: {
                            type: Scratch.ArgumentType.NUMBER,
                            defaultValue: 1
                        }
                    }
                },
                {
                    opcode: 'getImageResolution',
                    blockType: Scratch.BlockType.REPORTER,
                    text: 'getImageResolution [URI]',
                    arguments: {
                        URI: {
                            type: Scratch.ArgumentType.STRING,
                            defaultValue: "sans"
                        }
                    }
                },
            ],
            menus:{
                blendParam: {
                    acceptReporters: true,
                    items: [
                        {
                            text: 'None',
                            value: BlendParam.None
                        },
                        {
                            text: 'Screen',
                            value: BlendParam.Screen
                        },
                        {
                            text: 'LinearDodge',
                            value: BlendParam.LinearDodge
                        },
                        {
                            text: 'Overlay',
                            value: BlendParam.Overlay
                        },
                        {
                            text: 'SoftLight',
                            value: BlendParam.SoftLight
                        },
                        {
                            text: 'Difference',
                            value: BlendParam.Difference
                        },
                        {
                            text: 'Multiply',
                            value: BlendParam.Multiply
                        }
                    ]
                },
                maskingParam: {
                    acceptReporters: true,
                    items: [
                        {
                            text: 'None',
                            value: MaskingParam.None
                        },
                        {
                            text: 'VisibleInsideMask',
                            value: MaskingParam.VisibleInsideMask
                        },
                        {
                            text: 'VisibleOutsideMask',
                            value: MaskingParam.VisibleOutsideMask
                        },
                        {
                            text: 'Mask',
                            value: MaskingParam.Mask
                        }
                    ]
                }
            }
            };
        }
        savePenStencil(){
            saveStencilBufferIntoTexture(penTexture[0]);
        }
        restorePenStencil(){
            restoreStencilBufferFromTexture(penTexture[0]);
        }
        disableAutoPenScreenFilterApplying(){
            autoPenScreenFilter = false;
        }
        updatePenScreenManually(){
            renderer.updatePenScreen();
            renderer.dirty = true;
        }
        blendModeToNum(MODE){
            switch (MODE) {
                case BlendParam.None:
                    return 0.0;
                case BlendParam.Screen:
                    return 1.0;
                case BlendParam.LinearDodge:
                    return 2.0;
                case BlendParam.Overlay:
                    return 3.0;
                case BlendParam.SoftLight:
                    return 4.0;
                case BlendParam.Difference:
                    return 5.0;
                case BlendParam.Multiply:
                    return 6.0;
            }
            return 0;
        }
        maskingModeToNum(MODE){
            switch (MODE) {
                case MaskingParam.None:
                    return 0.0;
                case MaskingParam.VisibleInsideMask:
                    return 1.0;
                case MaskingParam.VisibleOutsideMask:
                    return 2.0;
                case MaskingParam.Mask:
                    return 3.0;
            }
            return 0;
        }
        setBlendMode(args, util){
            var renderedTarget = util.target;
            renderedTarget.blendMode = args.MODE;
            var drawable = renderedTarget.renderer._allDrawables[renderedTarget.drawableID];
            if (drawable){
                drawable._uniforms.u_blendMode = this.blendModeToNum(args.MODE);
            }
            
            if (renderedTarget.visible) {
                renderedTarget.emit('EVENT_TARGET_VISUAL_CHANGE', renderedTarget);
                renderer.dirty = true;
            }
        }
        setMaskingMode(args, util){
            var renderedTarget = util.target;
            renderedTarget.maskingMode = args.MODE;
            var drawable = renderedTarget.renderer._allDrawables[renderedTarget.drawableID];
            if (drawable){
                drawable._uniforms.u_maskingMode = this.maskingModeToNum(args.MODE);
            }
            
            if (renderedTarget.visible) {
                renderedTarget.emit('EVENT_TARGET_VISUAL_CHANGE', renderedTarget);
                renderer.dirty = true;
            }
        }
        clearMaskInPenScreen(args, util){
            twgl.bindFramebufferInfo(gl, penTexture[0]); 
            gl.clearStencil(0);
            gl.stencilMask(0xff);
            gl.clear(gl.STENCIL_BUFFER_BIT);
            renderer.dirty = true;
        }
        setBloom({intensity, bright, colorR, colorG, colorB}){
            u_intensity = intensity/100;
            u_brightnessThreshold = bright/100;
            u_color = [colorR/255, colorG/255, colorB/255];
            renderer.dirty = true;


        }
        setGrayscale({value}){
            desaturationFactor2 = value/100;
            renderer.dirty = true;
        }
        setGaussianBlur({value}){
            u_distortion = Math.max(0.10001, value/100);
            renderer.dirty = true;
        }
        setBlurryBlur({value}){
            u_blurryBlurDistortion = value/50;
            renderer.dirty = true;
        }
        setBlurFocus({value}){
            var a = Math.max(0.10001, value/100);
            u_focusBlurSize = Math.round(a/0.2)*0.2;
            renderer.dirty = true;
        }
        setAberration({value}){
            aberration_degree = (value-50)/2500;
            renderer.dirty = true;
        }
        fisheye({fisheye}){
            u_fisheye = fisheye/100;
            renderer.dirty = true;
        }
        setTiling(args, util) {
            var target = util.target;
            
            target.tileX = args.X;
            target.tileY = args.Y;
            var drawable = target.renderer._allDrawables[target.drawableID];

            if (drawable){
                drawable._uniforms.u_tileX = args.X;
                drawable._uniforms.u_tileY = args.Y;
            }
            
            if (target.visible) {
                target.emit('EVENT_TARGET_VISUAL_CHANGE', target);
                renderer.dirty = true;
            }
            
            
            
        }
        setScreenTiling(args, util){
            screenTileX = args.X;
            screenTileY = args.Y;
            renderer.dirty = true;
        }
        tint(args, util) {
            var target = util.target;
            this.setTint(target, args.R, args.G, args.B,args.A);
        }
        setTint (renderedTarget, R,G,B,A) {
            renderedTarget.tintR = R;
            renderedTarget.tintG = G;
            renderedTarget.tintB = B;
            renderedTarget.tintA = A;
            var drawable = renderedTarget.renderer._allDrawables[renderedTarget.drawableID];

            if (drawable){
                drawable._uniforms.u_tintR = 1 - R/255.0;
                drawable._uniforms.u_tintG = 1 - G/255.0;
                drawable._uniforms.u_tintB = 1 - B/255.0;
                drawable._uniforms.u_tintA = 1 - A/255.0;
            }
            
            if (renderedTarget.visible) {
                renderedTarget.emit('EVENT_TARGET_VISUAL_CHANGE', renderedTarget);
                renderer.dirty = true;
            }
        }
        getTint (renderedTarget){
            var rgb = ((renderedTarget.tintR&0x0ff)<<16)|((renderedTarget.tintG&0x0ff)<<8)|(renderedTarget.tintB&0x0ff);
            return rgb;
        }
        getImageResolution({URI}) {
            return new Promise(async function(resolve, reject) {
                if (!(await Scratch.canFetch(URI))) return reject();
                // eslint-disable-next-line no-restricted-syntax
                const image = new Image();
                image.onload = function() {
                    resolve(image.width.toString() + " " +image.height.toString());
                }
                image.src = URI;
            });
            

        }
    }

    Scratch.extensions.register(new Colorify());
})(Scratch);