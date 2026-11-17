(async () => {
  const list = document.getElementById('publication-list');
  if (!list) return;

  const el = (tag, className, value) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (value !== undefined) node.textContent = value;
    return node;
  };

  const link = (label, href, className) => {
    const anchor = el('a', className, label);
    anchor.href = href;
    anchor.target = '_blank';
    anchor.rel = 'noopener noreferrer';
    return anchor;
  };

  function renderAuthors(parts) {
    const byline = el('p', 'publication-authors');
    parts.forEach((part, index) => {
      if (index) byline.append(', ');
      if (part === 'Cifuentes') {
        byline.append(el('strong', 'publication-author--cifuentes', part));
      } else if (part === 'et al. (incl. Cifuentes)') {
        byline.append('et al. (incl. ', el('strong', 'publication-author--cifuentes', 'Cifuentes'), ')');
      } else {
        byline.append(part);
      }
    });
    return byline;
  }

  function render(record) {
    const article = el('article', 'publication-card');
    article.dataset.year = record.year;
    article.dataset.authorship = record.author_position === 1 ? 'authored' : 'coauthored';
    article.dataset.publicationType = record.category || 'articles';
    const year = el('span', 'publication-year');
    year.setAttribute('aria-label', `Published in ${record.year}`);
    const body = el('div', 'publication-card-body');
    body.append(el('h3', null, record.title), renderAuthors(record.byline));

    const details = el('div', 'publication-details');
    const venue = [record.journal, record.volume, record.page].filter(Boolean).join(', ');
    details.append(el('span', 'publication-venue', venue));
    details.append(link('ADS', `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(record.bibcode)}/abstract`, 'publication-link-badge'));
    if (record.arxiv) {
      details.append(link('arXiv', `https://arxiv.org/abs/${encodeURIComponent(record.arxiv)}`, 'publication-link-badge'));
    }
    const count = record.citation_count;
    const showCitations = record.category !== 'catalogues' && record.category !== 'proceedings';
    if (showCitations && Number.isInteger(count) && count >= 0) {
      const citations = link('', `https://ui.adsabs.harvard.edu/abs/${encodeURIComponent(record.bibcode)}/citations`, 'publication-citations');
      citations.append(el('span', 'publication-citation-count', count.toLocaleString('en-GB')),
        ` ${count === 1 ? 'citation' : 'citations'}`);
      details.append(citations);
    }
    body.append(details);
    article.append(year, body);
    return article;
  }

  try {
    const response = await fetch('assets/data/ads-publications.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('ADS cache unavailable');
    const data = await response.json();
    if (!Array.isArray(data.records) || !data.records.length) throw new Error('ADS cache empty');
    list.replaceChildren(...data.records.map(render));

    const note = document.getElementById('citation-source');
    const updated = new Date(data.updated_at);
    if (note && !Number.isNaN(updated.getTime())) {
      const time = el('time', null, new Intl.DateTimeFormat('en-GB', {
        day: 'numeric', month: 'long', year: 'numeric', timeZone: 'UTC'
      }).format(updated));
      time.dateTime = data.updated_at;
      note.replaceChildren('Articles and citation counts from NASA ADS · Updated ', time);
      note.hidden = false;
    }

    const typeButtons = [...document.querySelectorAll('[data-publication-type]')];
    const authorshipButtons = [...document.querySelectorAll('[data-publication-authorship]')];
    const empty = document.getElementById('publication-empty');
    let publicationType = 'articles';
    let authorship = 'all';

    function applyFilters() {
      typeButtons.forEach(button => {
        const active = button.dataset.publicationType === publicationType;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      authorshipButtons.forEach(button => {
        const active = button.dataset.publicationAuthorship === authorship;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      let lastYear = null;
      let visible = 0;
      for (const card of list.children) {
        card.hidden = card.dataset.publicationType !== publicationType ||
          (authorship !== 'all' && card.dataset.authorship !== authorship);
        const year = card.querySelector('.publication-year');
        if (card.hidden) continue;
        visible += 1;
        const showYear = card.dataset.year !== lastYear;
        year.textContent = showYear ? card.dataset.year : '';
        year.classList.toggle('publication-year--empty', !showYear);
        lastYear = card.dataset.year;
      }
      if (empty) {
        empty.hidden = visible > 0;
        empty.textContent = `No ${publicationType} match this selection in the current ADS record.`;
      }
    }
    typeButtons.forEach(button => button.addEventListener('click', () => {
      publicationType = button.dataset.publicationType;
      applyFilters();
    }));
    authorshipButtons.forEach(button => button.addEventListener('click', () => {
      authorship = button.dataset.publicationAuthorship;
      applyFilters();
    }));
    applyFilters();
  } catch {
    list.replaceChildren(el('p', 'publication-unavailable', 'The article list is temporarily unavailable. Please use the ADS link above.'));
  }
})();
