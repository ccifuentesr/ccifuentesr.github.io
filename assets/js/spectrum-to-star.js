const stepTabs = [...document.querySelectorAll('.kobe-step-tab')];
const stepPanels = [...document.querySelectorAll('.kobe-step-panel')];

function activateStep(tab, focus = false) {
  stepTabs.forEach(item => {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
  });
  stepPanels.forEach(panel => {
    panel.hidden = panel.id !== tab.getAttribute('aria-controls');
  });
  if (focus) {
    tab.focus({ preventScroll: true });
    tab.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'instant' });
  }
}

stepTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateStep(tab));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % stepTabs.length;
    if (event.key === 'ArrowLeft') next = (index - 1 + stepTabs.length) % stepTabs.length;
    if (event.key === 'Home') next = 0;
    if (event.key === 'End') next = stepTabs.length - 1;
    if (next === undefined) return;
    event.preventDefault();
    activateStep(stepTabs[next], true);
  });
});
