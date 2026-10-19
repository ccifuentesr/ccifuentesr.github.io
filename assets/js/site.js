document.getElementById('print-cv')?.addEventListener('click', () => window.print());
const musicPanels = document.querySelectorAll('[data-music-panel]');
function showMusicPanel() {
  const requested = location.hash.slice(1) || 'about';
  const name = [...musicPanels].some(p => p.dataset.musicPanel === requested) ? requested : 'about';
  musicPanels.forEach(p => { p.hidden = p.dataset.musicPanel !== name; });
  document.querySelectorAll('.collection-nav a').forEach(a => {
    if (a.hash === '#' + name) a.setAttribute('aria-current', 'page');
    else a.removeAttribute('aria-current');
  });
}
if (musicPanels.length) { showMusicPanel(); window.addEventListener('hashchange', showMusicPanel); }
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
