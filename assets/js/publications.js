(async () => {
  const links = [...document.querySelectorAll('[data-ads-bibcode]')];
  if (!links.length) return;
  try {
    const response = await fetch('assets/data/ads-citations.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('Citation cache unavailable');
    const data = await response.json();
    const updated = new Date(data.updated_at);
    if (Number.isNaN(updated.getTime())) throw new Error('Invalid update date');
    const counts = links.map(link => data.records?.[link.dataset.adsBibcode]?.citation_count);
    if (counts.some(count => !Number.isInteger(count) || count < 0)) {
      throw new Error('Incomplete citation cache');
    }
    links.forEach((link, index) => {
      const count = counts[index];
      link.textContent = `${count.toLocaleString('en-GB')} ${count === 1 ? 'citation' : 'citations'}`;
      link.hidden = false;
    });
    const note = document.getElementById('citation-source');
    if (note) {
      const date = document.createElement('time');
      date.dateTime = data.updated_at;
      date.textContent = new Intl.DateTimeFormat('en-GB', {
        day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC'
      }).format(updated);
      note.replaceChildren('Citation counts from NASA ADS · Updated ', date);
      note.hidden = false;
    }
  } catch {
    // Articles and their links remain usable; unavailable counts are never shown as zero.
  }
})();
