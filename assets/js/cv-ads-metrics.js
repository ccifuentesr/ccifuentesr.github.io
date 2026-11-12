(() => {
  const fields = [...document.querySelectorAll('[data-ads-metric]')];
  if (!fields.length) return;

  fetch('assets/data/ads-citations.json', { cache: 'no-store' })
    .then(response => {
      if (!response.ok) throw new Error('ADS metrics unavailable');
      return response.json();
    })
    .then(data => {
      const metrics = data.author_metrics;
      const values = fields.map(field => metrics?.[field.dataset.adsMetric]);
      if (values.some(value => !Number.isInteger(value) || value < 0)) {
        throw new Error('Invalid ADS metrics');
      }
      fields.forEach((field, index) => {
        field.textContent = values[index].toLocaleString('en-GB');
      });
    })
    .catch(() => {
      // Keep the readable fallback figures if ADS is temporarily unavailable.
    });
})();
