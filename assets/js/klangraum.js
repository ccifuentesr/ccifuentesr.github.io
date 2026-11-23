(() => {
  const canvas = document.getElementById('klangraum-journey');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const palette = ['70,112,226', '57,183,137', '204,73,91', '137,83,214', '67,176,205'];
  const points = Array.from({ length: 260 }, () => ({
    x: Math.random(), y: Math.random(),
    phase: Math.random() * Math.PI * 2,
    size: .2 + Math.random() * .25,
    opacity: .14 + Math.random() * .2
  }));
  const particles = Array.from({ length: 150 }, (_, i) => ({
    x: .04 + Math.random() * .92, y: .04 + Math.random() * .92,
    phase: Math.random() * Math.PI * 2,
    period: 3.2 + Math.random() * 4.8,
    size: i % 20 === 0 ? .65 + Math.random() * .35 : .3 + Math.random() * .45,
    color: palette[i % palette.length]
  }));
  let width = 0, height = 0, frame = 0, last = 0, time = 0, arrival = 0;
  let roomOpen = false;
  const active = () => location.hash === '#klangraum' && !document.hidden;

  function resize() {
    width = window.innerWidth;
    height = window.innerHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    if (active()) draw(0);
  }

  function haze(x, y, radius, color, opacity) {
    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
    gradient.addColorStop(0, `rgba(${color},${opacity})`);
    gradient.addColorStop(1, `rgba(${color},0)`);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
  }

  function drawParticle(x, y, radius, color, opacity) {
    const glow = ctx.createRadialGradient(x, y, 0, x, y, radius);
    glow.addColorStop(0, `rgba(${color},${opacity})`);
    glow.addColorStop(.35, `rgba(${color},${opacity * .48})`);
    glow.addColorStop(1, `rgba(${color},0)`);
    ctx.fillStyle = glow;
    ctx.beginPath();
    for (let i = 0; i <= 16; i++) {
      const angle = i / 16 * Math.PI * 2;
      const wobble = .78 + .2 * Math.sin(angle * 3 + x * .01) + .08 * Math.cos(angle * 5 + y * .012);
      const px = x + Math.cos(angle) * radius * wobble;
      const py = y + Math.sin(angle) * radius * wobble * (.68 + .18 * Math.sin(angle * 2));
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.closePath();
    ctx.fill();
  }

  function draw(dt) {
    time += dt;
    arrival += dt;
    const span = Math.max(width, height);
    ctx.fillStyle = '#07090c';
    ctx.fillRect(0, 0, width, height);
    // Overlapping aurora veils travel across the whole sky in varied directions.
    const auroras = [
      { color: '48,105,235', phase: .3, speed: .11, x: .18, y: .28, opacity: .125 },
      { color: '43,190,126', phase: 2.4, speed: -.085, x: .52, y: .57, opacity: .11 },
      { color: '218,62,82', phase: 4.1, speed: .072, x: .82, y: .38, opacity: .095 },
      { color: '132,70,213', phase: 1.5, speed: -.061, x: .36, y: .74, opacity: .085 }
    ];
    for (const aurora of auroras) {
      const drift = Math.sin(time * aurora.speed + aurora.phase);
      const cross = Math.cos(time * aurora.speed * .63 + aurora.phase * 1.7);
      haze(width * (aurora.x + drift * .34), height * (aurora.y + cross * .23), span * .78, aurora.color, aurora.opacity);
      haze(width * (aurora.x - drift * .2), height * (aurora.y + cross * .12), span * .46, aurora.color, aurora.opacity * .55);
    }

    // Fixed-size points drift locally; there is no depth, zoom or vanishing point.
    for (const point of points) {
      const x = point.x * width + 30 * Math.sin(time * .09 + point.phase) + 9 * Math.sin(time * .17 + point.phase * 1.7);
      const y = point.y * height + 42 * Math.cos(time * .075 + point.phase) + 12 * Math.sin(time * .13 + point.phase * 1.4);
      const softness = Math.pow((Math.sin(time * .72 + point.phase) + 1) / 2, 2);
      drawParticle(x, y, point.size * 4.2, '197,207,216', point.opacity * softness);
    }
    for (const particle of particles) {
      const phase = time * Math.PI * 2 / particle.period + particle.phase;
      const opacity = .38 * Math.pow((Math.sin(phase) + 1) / 2, 5);
      const x = particle.x * width + 58 * Math.sin(time * .09 + particle.phase) + 16 * Math.sin(time * .11 + particle.phase * 1.9);
      const y = particle.y * height + 76 * Math.cos(time * .08 + particle.phase) + 19 * Math.sin(time * .09 + particle.phase * 1.3);
      drawParticle(x, y, particle.size * 5.4, particle.color, opacity * .9);
    }
    ctx.shadowBlur = 0;
    const vignette = ctx.createRadialGradient(width / 2, height * .45, 0, width / 2, height * .45, Math.max(width, height) * .68);
    vignette.addColorStop(.35, 'rgba(0,0,0,0)');
    vignette.addColorStop(1, 'rgba(0,0,0,.64)');
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, width, height);
    // Every entrance begins in black, then reveals the scene over nine seconds.
    const progress = reduced.matches ? 1 : Math.min(1, arrival / 9);
    const reveal = progress * progress * (3 - 2 * progress);
    ctx.fillStyle = `rgba(0,0,0,${1 - reveal})`;
    ctx.fillRect(0, 0, width, height);
  }

  function tick(now) {
    frame = 0;
    if (!active()) return;
    draw(last ? Math.min((now - last) / 1000, .05) : 0);
    last = now;
    if (!reduced.matches) frame = requestAnimationFrame(tick);
  }
  function sync() {
    cancelAnimationFrame(frame);
    frame = 0;
    last = 0;
    const entering = location.hash === '#klangraum';
    if (entering && !roomOpen) arrival = 0;
    roomOpen = entering;
    if (active()) { resize(); frame = requestAnimationFrame(tick); }
  }
  window.addEventListener('resize', resize);
  window.addEventListener('hashchange', sync);
  document.addEventListener('visibilitychange', sync);
  reduced.addEventListener('change', sync);
  sync();
})();
