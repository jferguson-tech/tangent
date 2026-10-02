// WebGL2 viewport: shows the baked maps under a view transform, lit or raw.
// Lit and emissive views render into a half-float target, get bloom, then are
// tone-mapped. Raw map views (normal, height, ...) are shown untouched.

const VS = `#version 300 es
in vec2 a_pos;
uniform vec2 u_canvasM;   // canvas size in meters
uniform vec2 u_offset;    // screen offset of the canvas origin (css px)
uniform float u_scale;    // css px per meter
uniform vec2 u_viewport;  // css px
uniform float u_tiles;
uniform float u_detail;   // 1 = drawing the zoomed-in detail patch
uniform vec4 u_quad;      // detail patch rectangle in canvas uv (x0, y0, x1, y1)
out vec2 v_uv;            // canvas uv (outside 0..1 = repeated tiles)
out vec2 v_map;           // detail texture uv
void main() {
  vec2 uv;
  if (u_detail > 0.5) {
    uv = mix(u_quad.xy, u_quad.zw, a_pos);
    v_map = a_pos;
  } else {
    float k = (u_tiles - 1.0) * 0.5;
    uv = mix(vec2(-k), vec2(1.0 + k), a_pos);
    v_map = vec2(0.0);
  }
  v_uv = uv;
  vec2 s = u_offset + uv * u_canvasM * u_scale;
  vec2 c = s / u_viewport * 2.0 - 1.0;
  gl_Position = vec4(c.x, -c.y, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
in vec2 v_map;
uniform float u_detail;
uniform sampler2D u_normal, u_height, u_ao, u_curv, u_id, u_emis, u_spill, u_wires;
uniform vec3 u_wireAlbedo;
uniform float u_wireMetal, u_wireRough;
uniform int u_mode;
uniform vec2 u_canvasM;
uniform vec2 u_hrange;
uniform int u_nlights;
uniform vec3 u_lpos[3];
uniform vec3 u_ldir[3];
uniform vec3 u_lcol[3];
uniform vec3 u_lcone[3];   // cos inner, cos outer, directional flag
uniform vec3 u_albedo;
uniform float u_metal, u_rough;
uniform vec3 u_ambient;
uniform float u_hasAO;
uniform sampler2D u_env0, u_env1, u_env2, u_env3, u_env4, u_irr;
uniform float u_useEnv, u_envInt, u_envRot;
uniform float u_emisScale, u_spillScale, u_spillGain;
uniform vec2 u_spillTexel;
uniform float u_tonemap;   // 1 = no HDR target available: tone-map here
out vec4 o;

const float PI = 3.14159265;
const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);

vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

vec3 brdf(vec3 N, vec3 V, vec3 L, vec3 base, float metal, float rough) {
  vec3 H = normalize(V + L);
  float nol = max(dot(N, L), 0.0), nov = max(dot(N, V), 1e-4);
  float noh = max(dot(N, H), 0.0), voh = max(dot(V, H), 1e-4);
  float a = max(rough * rough, 0.002), a2 = a * a;
  float d = noh * noh * (a2 - 1.0) + 1.0;
  float D = a2 / (PI * d * d);
  float g1v = 2.0 * nov / (nov + sqrt(a2 + (1.0 - a2) * nov * nov));
  float g1l = 2.0 * nol / (nol + sqrt(a2 + (1.0 - a2) * nol * nol) + 1e-6);
  vec3 F0 = mix(vec3(0.04), base, metal);
  vec3 F = F0 + (1.0 - F0) * pow(1.0 - voh, 5.0);
  vec3 spec = D * g1v * g1l * F / max(4.0 * nov * nol, 1e-4);
  vec3 diff = (1.0 - F) * (1.0 - metal) * base / PI;
  return (spec + diff) * nol;
}

// Equirect lookup: row 0 = zenith, world z up, azimuth from +x toward +y.
vec2 envUV(vec3 d) {
  float phi = atan(d.y, d.x) - u_envRot;
  return vec2(fract(0.5 + phi / (2.0 * PI)), acos(clamp(d.z, -1.0, 1.0)) / PI);
}

vec3 envLevel(int i, vec2 uv) {
  if (i <= 0) return textureLod(u_env0, uv, 0.0).rgb;
  if (i == 1) return textureLod(u_env1, uv, 0.0).rgb;
  if (i == 2) return textureLod(u_env2, uv, 0.0).rgb;
  if (i == 3) return textureLod(u_env3, uv, 0.0).rgb;
  return textureLod(u_env4, uv, 0.0).rgb;
}

vec3 envSpec(vec3 R, float rough) {
  vec2 uv = envUV(R);
  float f = clamp(rough, 0.0, 1.0) * 4.0;
  int i = int(floor(f));
  return mix(envLevel(i, uv), envLevel(min(i + 1, 4), uv), f - float(i));
}

vec2 envBRDF(float nov, float rough) {
  vec4 r = rough * vec4(-1.0, -0.0275, -0.572, 0.022) + vec4(1.0, 0.0425, 1.04, -0.04);
  float a004 = min(r.x * r.x, exp2(-9.28 * nov)) * r.x + r.y;
  return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Emission and spill are stored as normalized sRGB; decode to linear radiance.
vec3 emission(vec2 uv) { return pow(texture(u_emis, uv).rgb, vec3(2.2)) * u_emisScale; }
vec3 spill(vec2 uv) { return pow(texture(u_spill, uv).rgb, vec3(2.2)) * u_spillScale; }

void main() {
  vec2 cuv = fract(v_uv);                        // canvas uv (spill is canvas-wide)
  vec2 uv = u_detail > 0.5 ? v_map : cuv;        // uv into the bound map textures
  bool center = all(greaterThanEqual(v_uv, vec2(0.0))) && all(lessThan(v_uv, vec2(1.0)));
  vec3 c;
  bool hdr = false;
  if (u_mode == 1) c = texture(u_normal, uv).rgb;
  else if (u_mode == 2) c = texture(u_height, uv).rrr;
  else if (u_mode == 3) c = texture(u_ao, uv).rrr;
  else if (u_mode == 4) c = texture(u_curv, uv).rrr;
  else if (u_mode == 5) c = texture(u_id, uv).rgb;
  else if (u_mode == 6) { c = emission(uv); hdr = true; }
  else {
    hdr = true;
    vec3 N = normalize(texture(u_normal, uv).rgb * 2.0 - 1.0);
    float wm = step(0.5, texture(u_wires, uv).r);   // wires use their own material
    vec3 alb = mix(u_albedo, u_wireAlbedo, wm);
    float met = mix(u_metal, u_wireMetal, wm), rgh = mix(u_rough, u_wireRough, wm);
    float h = mix(u_hrange.x, u_hrange.y, texture(u_height, uv).r);
    vec3 P = vec3((v_uv.x - 0.5) * u_canvasM.x, (0.5 - v_uv.y) * u_canvasM.y, h);
    vec3 V = vec3(0.0, 0.0, 1.0);
    vec3 col = vec3(0.0);
    for (int i = 0; i < 3; i++) {
      if (i >= u_nlights) break;
      vec3 L; vec3 rad;
      if (u_lcone[i].z > 0.5) {
        L = normalize(u_lpos[i]);
        rad = u_lcol[i];
      } else {
        vec3 Lv = u_lpos[i] - P;
        float dist = length(Lv);
        L = Lv / dist;
        float cosA = dot(-L, u_ldir[i]);
        float spot = smoothstep(u_lcone[i].y, u_lcone[i].x, cosA);
        rad = u_lcol[i] * spot / (dist * dist);
      }
      col += brdf(N, V, L, alb, met, rgh) * rad;
    }
    float ao = mix(1.0, texture(u_ao, uv).r, u_hasAO);
    if (u_useEnv > 0.5) {
      float nov = max(dot(N, V), 1e-4);
      vec3 R = reflect(-V, N);
      vec3 F0 = mix(vec3(0.04), alb, met);
      vec2 ab = envBRDF(nov, rgh);
      vec3 spec = envSpec(R, rgh) * (F0 * ab.x + ab.y);
      vec3 diff = textureLod(u_irr, envUV(N), 0.0).rgb * alb * (1.0 - met);
      col += (spec + diff) * u_envInt * ao;
    } else {
      col += u_ambient * alb * (0.6 + 0.4 * N.z) * ao;
    }
    // Fake light spill: blurred emission as a nearby light. Its gradient points
    // toward the source, which tilts the light direction so facing bevels catch it.
    if (u_spillScale > 0.0 && u_spillGain > 0.0) {
      vec3 sp = spill(cuv);
      float l0 = dot(sp, LUMA);
      if (l0 > 1e-5) {
        vec2 d = u_spillTexel * 3.0;
        float gx = dot(spill(cuv + vec2(d.x, 0.0)) - spill(cuv - vec2(d.x, 0.0)), LUMA);
        float gy = dot(spill(cuv - vec2(0.0, d.y)) - spill(cuv + vec2(0.0, d.y)), LUMA);
        vec2 g = vec2(gx, gy) / (2.0 * l0);
        vec3 L = normalize(vec3(g * 4.0, 1.0));
        col += brdf(N, V, L, alb, met, max(rgh, 0.55)) * sp * u_spillGain * PI * ao;
      }
    }
    col += emission(uv);
    c = col;
  }
  if (!center) c *= 0.5;
  if (hdr && u_tonemap > 0.5) c = pow(aces(c), vec3(1.0 / 2.2));
  o = vec4(c, 1.0);
}`;

// ---- bloom / composite ------------------------------------------------------
const VS_FULL = `#version 300 es
out vec2 v_uv;
void main() {
  vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
  v_uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FS_PREFILTER = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;       // source texel size
uniform float u_threshold;
out vec4 o;
void main() {
  vec4 s = texture(u_src, v_uv + u_texel * vec2(-0.5, -0.5)) + texture(u_src, v_uv + u_texel * vec2(0.5, -0.5))
         + texture(u_src, v_uv + u_texel * vec2(-0.5, 0.5)) + texture(u_src, v_uv + u_texel * vec2(0.5, 0.5));
  vec3 c = s.rgb * 0.25;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  float knee = max(u_threshold * 0.5, 1e-4);
  float soft = clamp(l - u_threshold + knee, 0.0, 2.0 * knee);
  soft = soft * soft / (4.0 * knee);
  float w = max(soft, l - u_threshold) / max(l, 1e-5);
  o = vec4(c * max(w, 0.0), 1.0);
}`;

const FS_DOWN = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;
out vec4 o;
void main() {
  vec3 a = texture(u_src, v_uv + u_texel * vec2(-1.0, -1.0)).rgb;
  vec3 b = texture(u_src, v_uv + u_texel * vec2(1.0, -1.0)).rgb;
  vec3 c = texture(u_src, v_uv + u_texel * vec2(-1.0, 1.0)).rgb;
  vec3 d = texture(u_src, v_uv + u_texel * vec2(1.0, 1.0)).rgb;
  vec3 e = texture(u_src, v_uv).rgb;
  o = vec4(e * 0.5 + (a + b + c + d) * 0.125, 1.0);
}`;

const FS_UP = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;
uniform float u_radius;
out vec4 o;
void main() {
  vec2 t = u_texel * u_radius;
  vec3 s = texture(u_src, v_uv).rgb * 4.0;
  s += (texture(u_src, v_uv + vec2(t.x, 0.0)).rgb + texture(u_src, v_uv - vec2(t.x, 0.0)).rgb
      + texture(u_src, v_uv + vec2(0.0, t.y)).rgb + texture(u_src, v_uv - vec2(0.0, t.y)).rgb) * 2.0;
  s += texture(u_src, v_uv + t).rgb + texture(u_src, v_uv - t).rgb
     + texture(u_src, v_uv + vec2(t.x, -t.y)).rgb + texture(u_src, v_uv + vec2(-t.x, t.y)).rgb;
  o = vec4(s / 16.0, 1.0);
}`;

const FS_COMPOSITE = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_scene, u_bloom;
uniform float u_intensity, u_hdr, u_useBloom;
out vec4 o;
vec3 aces(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
void main() {
  vec4 s = texture(u_scene, v_uv);
  if (u_hdr < 0.5) { o = s; return; }
  vec3 b = u_useBloom > 0.5 ? texture(u_bloom, v_uv).rgb * u_intensity : vec3(0.0);
  vec3 c = pow(aces(s.rgb + b), vec3(1.0 / 2.2));
  float glow = clamp(dot(pow(aces(b), vec3(1.0 / 2.2)), vec3(0.3333)) * 1.5, 0.0, 1.0);
  // Outside the canvas the glow still shows over the checkerboard.
  o = vec4(s.a > 0.0 ? c : pow(aces(b), vec3(1.0 / 2.2)), max(s.a, glow));
}`;

const MODE = { lit: 0, normal: 1, height: 2, ao: 3, curvature: 4, id: 5, emissive: 6 };
const UNITS = { normal: 0, height: 1, ao: 2, curvature: 3, id: 4, emissive: 11, spill: 12, wires: 13 };
const BLOOM_LEVELS = 6;

export class GLView {
  constructor(canvas) {
    this.canvas = canvas;
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.hdr = !!gl.getExtension('EXT_color_buffer_float');
    this.prog = this._program(VS, FS);
    this.u = this._uniforms(this.prog);
    if (this.hdr) {
      this.pPre = this._program(VS_FULL, FS_PREFILTER);
      this.pDown = this._program(VS_FULL, FS_DOWN);
      this.pUp = this._program(VS_FULL, FS_UP);
      this.pComp = this._program(VS_FULL, FS_COMPOSITE);
      this.uPre = this._uniforms(this.pPre);
      this.uDown = this._uniforms(this.pDown);
      this.uUp = this._uniforms(this.pUp);
      this.uComp = this._uniforms(this.pComp);
      this.emptyVao = gl.createVertexArray();
    }
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const loc = gl.getAttribLocation(this.prog, 'a_pos');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    this.tex = {};
    this.texSize = {};
    for (const name of Object.keys(UNITS)) {
      this.tex[name] = this._texture(name === 'normal' ? [128, 128, 255, 255] : name === 'emissive' || name === 'spill' || name === 'wires' ? [0, 0, 0, 255] : [128, 128, 128, 255]);
    }
    this.loaded = new Set();
    this.filterNearest = null;
    this.envTex = [];
    this.envId = null;
    this.targets = null;
    this.detail = null;      // { tex: {name: texture}, rect: [u0, v0, u1, v1], size: [w, h], token }
  }

  _uniforms(prog) {
    const gl = this.gl;
    const u = {};
    const n = gl.getProgramParameter(prog, gl.ACTIVE_UNIFORMS);
    for (let i = 0; i < n; i++) {
      const info = gl.getActiveUniform(prog, i);
      u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(prog, info.name);
    }
    return u;
  }

  // Environment levels from /api/env/gl: float16 RGBA, linearly filtered.
  setEnv(data) {
    const gl = this.gl;
    const decode = (b64) => {
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new Uint16Array(bytes.buffer);
    };
    const all = [...data.levels, data.irradiance];
    all.forEach((lv, i) => {
      if (!this.envTex[i]) this.envTex[i] = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.envTex[i]);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, lv.w, lv.h, 0, gl.RGBA, gl.HALF_FLOAT, decode(lv.data));
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    });
    this.envId = data.key;
  }

  _program(vs, fs) {
    const gl = this.gl;
    const mk = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, mk(gl.VERTEX_SHADER, vs));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    return p;
  }

  _texture(rgba) {
    const gl = this.gl;
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array(rgba));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    return t;
  }

  _target(w, h) {
    const gl = this.gl;
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const fb = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    const ok = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return ok ? { tex, fb, w, h } : null;
  }

  _ensureTargets() {
    if (!this.hdr) return null;
    const W = this.canvas.width, H = this.canvas.height;
    if (this.targets && this.targets.W === W && this.targets.H === H) return this.targets;
    const gl = this.gl;
    if (this.targets) {
      for (const t of [this.targets.scene, ...this.targets.mips]) { gl.deleteTexture(t.tex); gl.deleteFramebuffer(t.fb); }
    }
    const scene = this._target(W, H);
    const mips = [];
    let w = W, h = H;
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      w = Math.max(1, w >> 1); h = Math.max(1, h >> 1);
      const t = this._target(w, h);
      if (t) mips.push(t);
    }
    if (!scene || mips.length < BLOOM_LEVELS) { this.hdr = false; this.targets = null; return null; }
    this.targets = { W, H, scene, mips };
    return this.targets;
  }

  // maps: {name: dataURL}
  async setMaps(maps) {
    const gl = this.gl;
    const jobs = Object.entries(maps).filter(([k]) => k in UNITS).map(([name, url]) => new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res([name, img]);
      img.onerror = rej;
      img.src = url;
    }));
    const imgs = await Promise.all(jobs);
    for (const [name, img] of imgs) {
      gl.bindTexture(gl.TEXTURE_2D, this.tex[name]);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      this.texSize[name] = [img.width, img.height];
      this.loaded.add(name);
    }
    this.filterNearest = null;  // force filter refresh
  }

  // Detail patch: maps {name: dataURL} covering rect (canvas uv). `token`
  // lets the caller drop a result that arrives after the document changed.
  async setDetail(maps, rect, token) {
    const gl = this.gl;
    const entries = Object.entries(maps).filter(([k]) => k in UNITS && k !== 'spill');
    const imgs = await Promise.all(entries.map(([name, url]) => new Promise((res, rej) => {
      const img = new Image();
      img.onload = () => res([name, img]);
      img.onerror = rej;
      img.src = url;
    })));
    if (this.detailToken !== token) return false;     // superseded while decoding
    const tex = {};
    let size = [1, 1];
    for (const [name, img] of imgs) {
      const t = (this.detail && this.detail.tex[name]) || this._spare?.[name] || gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      tex[name] = t;
      size = [img.width, img.height];
    }
    this.detail = { tex, rect, size, token };
    return true;
  }

  clearDetail() {
    if (this.detail) this._spare = this.detail.tex;   // reuse the texture objects next time
    this.detail = null;
    this.detailToken = (this.detailToken || 0) + 1;
  }

  resize(w, h, dpr) {
    const W = Math.max(1, Math.round(w * dpr)), H = Math.max(1, Math.round(h * dpr));
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }
  }

  // p: {mode, canvasM, offset, scale, viewport, tiles, hrange, light, emis, bloom}
  draw(p) {
    const gl = this.gl;
    const hdrMode = p.mode === 'lit' || p.mode === 'emissive';
    const T = hdrMode ? this._ensureTargets() : null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, T ? T.scene.fb : null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    this._drawScene(p, !T, null);
    if (this.detail) this._drawScene(p, !T, this.detail);
    if (!T) return;
    const b = p.bloom || {};
    const useBloom = b.enabled !== false && (b.intensity ?? 0) > 0;
    gl.bindVertexArray(this.emptyVao);
    if (useBloom) this._bloom(T, b);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.pComp);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, T.scene.tex);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, T.mips[0].tex);
    gl.uniform1i(this.uComp.u_scene, 0);
    gl.uniform1i(this.uComp.u_bloom, 1);
    gl.uniform1f(this.uComp.u_intensity, (b.intensity ?? 0.2) / T.mips.length);
    gl.uniform1f(this.uComp.u_hdr, 1);
    gl.uniform1f(this.uComp.u_useBloom, useBloom ? 1 : 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  _bloom(T, b) {
    const gl = this.gl;
    const pass = (prog, u, src, dst, extra) => {
      gl.bindFramebuffer(gl.FRAMEBUFFER, dst.fb);
      gl.viewport(0, 0, dst.w, dst.h);
      gl.useProgram(prog);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src.tex);
      gl.uniform1i(u.u_src, 0);
      gl.uniform2f(u.u_texel, 1 / src.w, 1 / src.h);
      if (extra) extra();
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    };
    gl.disable(gl.BLEND);
    pass(this.pPre, this.uPre, T.scene, T.mips[0], () => gl.uniform1f(this.uPre.u_threshold, b.threshold ?? 1.5));
    for (let i = 1; i < T.mips.length; i++) pass(this.pDown, this.uDown, T.mips[i - 1], T.mips[i]);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = T.mips.length - 1; i > 0; i--) {
      pass(this.pUp, this.uUp, T.mips[i], T.mips[i - 1], () => gl.uniform1f(this.uUp.u_radius, b.radius ?? 1.0));
    }
    gl.disable(gl.BLEND);
  }

  _drawScene(p, tonemapHere, detail) {
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.bindVertexArray(this.vao);
    const u = this.u;
    if (detail) {
      // Detail pass: the same shading over just the patch, reading its textures.
      const [u0, v0, u1, v1] = detail.rect;
      const texelPx = (p.scale * p.canvasM[0] * (u1 - u0)) / detail.size[0];
      const f = texelPx > 3 ? gl.NEAREST : gl.LINEAR;
      for (const [name, unit] of Object.entries(UNITS)) {
        const t = name === 'spill' ? this.tex.spill : detail.tex[name] || this.tex[name];
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, t);
        if (name !== 'spill' && detail.tex[name]) {
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
        }
      }
      gl.uniform1f(u.u_detail, 1);
      gl.uniform4f(u.u_quad, u0, v0, u1, v1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.uniform1f(u.u_detail, 0);
      gl.bindVertexArray(null);
      return;
    }
    gl.uniform1f(u.u_detail, 0);
    const ts = this.texSize.normal || [1, 1];
    const texelPx = p.scale * p.canvasM[0] / ts[0];
    const nearest = texelPx > 3;
    if (nearest !== this.filterNearest) {
      for (const [name, t] of Object.entries(this.tex)) {
        gl.bindTexture(gl.TEXTURE_2D, t);
        // Spill is a smooth field: always filter it linearly.
        const f = nearest && name !== 'spill' ? gl.NEAREST : gl.LINEAR;
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, f);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, f);
      }
      this.filterNearest = nearest;
    }
    for (const [name, unit] of Object.entries(UNITS)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, this.tex[name]);
    }
    gl.uniform1i(u.u_normal, 0);
    gl.uniform1i(u.u_height, 1);
    gl.uniform1i(u.u_ao, 2);
    gl.uniform1i(u.u_curv, 3);
    gl.uniform1i(u.u_id, 4);
    gl.uniform1i(u.u_emis, 11);
    gl.uniform1i(u.u_spill, 12);
    gl.uniform1i(u.u_wires, 13);
    gl.uniform1i(u.u_mode, MODE[p.mode] ?? 0);
    gl.uniform2fv(u.u_canvasM, p.canvasM);
    gl.uniform2fv(u.u_offset, p.offset);
    gl.uniform1f(u.u_scale, p.scale);
    gl.uniform2fv(u.u_viewport, p.viewport);
    gl.uniform1f(u.u_tiles, p.tiles);
    gl.uniform2fv(u.u_hrange, p.hrange || [0, 0]);
    gl.uniform1f(u.u_tonemap, tonemapHere ? 1 : 0);
    const L = p.light;
    const n = Math.min(3, L.lights.length);
    const pos = new Float32Array(9), dir = new Float32Array(9), col = new Float32Array(9), cone = new Float32Array(9);
    L.lights.slice(0, 3).forEach((l, i) => {
      pos.set(l.pos, i * 3); dir.set(l.dir || [0, 0, -1], i * 3); col.set(l.color, i * 3);
      cone.set([l.cosInner ?? 1, l.cosOuter ?? 0, l.directional ? 1 : 0], i * 3);
    });
    gl.uniform1i(u.u_nlights, n);
    gl.uniform3fv(u.u_lpos, pos);
    gl.uniform3fv(u.u_ldir, dir);
    gl.uniform3fv(u.u_lcol, col);
    gl.uniform3fv(u.u_lcone, cone);
    gl.uniform3fv(u.u_albedo, L.albedo);
    gl.uniform1f(u.u_metal, L.metallic);
    gl.uniform1f(u.u_rough, L.roughness);
    gl.uniform3fv(u.u_ambient, L.ambient);
    const wmat = L.wire || { albedo: L.albedo, metallic: L.metallic, roughness: L.roughness };
    gl.uniform3fv(u.u_wireAlbedo, wmat.albedo);
    gl.uniform1f(u.u_wireMetal, wmat.metallic);
    gl.uniform1f(u.u_wireRough, wmat.roughness);
    gl.uniform1f(u.u_hasAO, this.loaded.has('ao') ? 1 : 0);
    const E = p.emis || {};
    gl.uniform1f(u.u_emisScale, this.loaded.has('emissive') ? (E.scale || 0) : 0);
    gl.uniform1f(u.u_spillScale, this.loaded.has('spill') ? (E.spillScale || 0) : 0);
    gl.uniform1f(u.u_spillGain, E.spill === false ? 0 : (E.spillGain ?? 1));
    const ss = this.texSize.spill || [1, 1];
    gl.uniform2f(u.u_spillTexel, 1 / ss[0], 1 / ss[1]);
    const env = L.env && this.envTex.length === 6 ? L.env : null;
    gl.uniform1f(u.u_useEnv, env ? 1 : 0);
    if (env) {
      ['u_env0', 'u_env1', 'u_env2', 'u_env3', 'u_env4', 'u_irr'].forEach((name, i) => {
        gl.activeTexture(gl.TEXTURE5 + i);
        gl.bindTexture(gl.TEXTURE_2D, this.envTex[i]);
        gl.uniform1i(u[name], 5 + i);
      });
      gl.uniform1f(u.u_envInt, env.intensity);
      gl.uniform1f(u.u_envRot, env.rotation * Math.PI / 180);
    } else {
      // Keep env samplers off the units used by the maps' different formats.
      ['u_env0', 'u_env1', 'u_env2', 'u_env3', 'u_env4', 'u_irr'].forEach((name, i) => gl.uniform1i(u[name], 5 + i));
    }
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
  }
}
