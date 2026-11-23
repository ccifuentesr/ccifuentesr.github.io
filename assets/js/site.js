document.getElementById('print-cv')?.addEventListener('click', () => window.print());

const themeToggle = document.querySelector('[data-theme-toggle]');
function syncTheme() {
  if (!themeToggle) return;
  const isLight = document.documentElement.dataset.theme === 'light';
  themeToggle.setAttribute('aria-pressed', String(isLight));
  themeToggle.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
  themeToggle.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
}
themeToggle?.addEventListener('click', () => {
  const useLight = document.documentElement.dataset.theme !== 'light';
  if (useLight) document.documentElement.dataset.theme = 'light';
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem('site-theme', useLight ? 'light' : 'dark'); } catch (error) {}
  syncTheme();
});
syncTheme();

const currentPage = location.pathname.split('/').pop() || 'index.html';
const activeFooterPage = currentPage.startsWith('echoes-') ? 'echoes.html' : currentPage;
document.querySelectorAll('footer nav a').forEach(link => {
  const linkPage = new URL(link.href, location.href).pathname.split('/').pop();
  if (linkPage === activeFooterPage) link.hidden = true;
});
const musicPanels = document.querySelectorAll('[data-music-panel]');
const musicBack = document.querySelector('[data-music-back]');
const roomAudio = document.getElementById('klangraum-audio');
function showMusicPanel() {
  const requested = location.hash.slice(1) || 'about';
  const name = [...musicPanels].some(p => p.dataset.musicPanel === requested) ? requested : 'about';
  musicPanels.forEach(p => { p.hidden = p.dataset.musicPanel !== name; });
  if (musicBack) musicBack.hidden = name === 'about';
  document.body.classList.toggle('klangraum-active', name === 'klangraum');
  if (roomAudio && name !== 'klangraum') { roomAudio.pause(); roomAudio.currentTime = 0; }
  document.querySelectorAll('.collection-nav a').forEach(a => {
    if (a.hash === '#' + name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}
if (musicPanels.length) { showMusicPanel(); window.addEventListener('hashchange', showMusicPanel); }
document.querySelector('.klangraum-button')?.addEventListener('click', () => {
  document.getElementById('global-audio')?.pause();
  roomAudio?.play().catch(() => {});
});
document.querySelector('.klangraum-space')?.addEventListener('pointerdown', () => {
  if (roomAudio?.paused) roomAudio.play().catch(() => {});
});
const listeningRoom = document.querySelector('.music-listening-room');
const klangraumButton = listeningRoom?.querySelector('.klangraum-button');
klangraumButton?.addEventListener('pointermove', event => {
  const rect = klangraumButton.getBoundingClientRect();
  klangraumButton.style.setProperty('--spot-x', `${event.clientX - rect.left}px`);
  klangraumButton.style.setProperty('--spot-y', `${event.clientY - rect.top}px`);
});
document.querySelectorAll('.tab').forEach(tab => {
  tab.setAttribute('role', 'button'); tab.tabIndex = 0;
  const activate = () => {
    const group = tab.closest('[data-music-panel]');
    group.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
    group.querySelectorAll('.tab-pane').forEach(p => { p.hidden = p.id !== tab.dataset.tab + '-content'; });
  };
  tab.addEventListener('click', activate);
  tab.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activate(); } });
});

const photoPanels = document.querySelectorAll('[data-photo-panel]');
function showPhotoPanel() {
  const requested = location.hash.slice(1);
  const name = [...photoPanels].some(p => p.dataset.photoPanel === requested) ? requested : 'light';
  photoPanels.forEach(p => { p.hidden = p.dataset.photoPanel !== name; });
  document.querySelectorAll('.collection-nav a').forEach(a => {
    if (a.hash === '#' + name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}
if (photoPanels.length) { showPhotoPanel(); window.addEventListener('hashchange', showPhotoPanel); }
