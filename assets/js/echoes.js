const echoesThemeToggle = document.getElementById('echoes-theme-toggle');

function syncEchoesTheme() {
  const isLight = document.documentElement.dataset.echoesTheme === 'light';
  echoesThemeToggle.setAttribute('aria-pressed', String(isLight));
  echoesThemeToggle.setAttribute('aria-label', isLight ? 'Switch to dark mode' : 'Switch to light mode');
  echoesThemeToggle.title = isLight ? 'Switch to dark mode' : 'Switch to light mode';
}

echoesThemeToggle.addEventListener('click', () => {
  const useLight = document.documentElement.dataset.echoesTheme !== 'light';
  if (useLight) document.documentElement.dataset.echoesTheme = 'light';
  else delete document.documentElement.dataset.echoesTheme;
  try { localStorage.setItem('echoes-theme', useLight ? 'light' : 'dark'); } catch (error) {}
  syncEchoesTheme();
});

syncEchoesTheme();
