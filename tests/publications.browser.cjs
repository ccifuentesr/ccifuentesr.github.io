// Run with PLAYWRIGHT_MODULE pointing to an installed playwright-core package.
// PUBLICATIONS_URL defaults to a local HTTP preview.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright-core');
const assert = require('node:assert/strict');

(async () => {
  const browser = await chromium.launch({ headless: true });
  try {
    for (const width of [1280, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 960 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(process.env.PUBLICATIONS_URL || 'http://127.0.0.1:8765/publications.html');
      await page.waitForSelector('.publication-card h3');
      const cards = page.locator('.publication-card');
      const total = await cards.count();
      // A type-button selector must never overwrite card.textContent.
      assert.equal(await page.locator('.publication-card h3').count(), total);
      const first = page.locator('.publication-card:not([hidden])').first();
      assert.equal(await first.locator('.publication-venue').evaluate(e => getComputedStyle(e).fontStyle), 'italic');
      assert.equal(await first.locator('.publication-link-badge').first().evaluate(e => getComputedStyle(e).borderTopStyle), 'solid');
      assert.equal(await first.locator('.publication-citation-count').count(), 1);
      assert.ok(parseFloat(await first.locator('h3').evaluate(e => getComputedStyle(e).fontSize)) > parseFloat(await first.locator('.publication-authors').evaluate(e => getComputedStyle(e).fontSize)));
      assert.ok((await first.locator('h3').boundingBox()).width > width * 0.5);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.locator('button[data-publication-authorship="authored"]').click();
      assert.ok(await page.locator('.publication-card:not([hidden])').count() > 0);
      await page.locator('button[data-publication-authorship="all"]').click();
      await page.locator('button[data-publication-type="catalogues"]').click();
      assert.equal(await page.locator('.publication-card:not([hidden]) .publication-citations').count(), 0);
      await page.locator('button[data-publication-type="articles"]').click();
      if (width < 641) {
        const row = await page.locator('.publication-filters').first().boundingBox();
        const authorRow = await page.locator('.publication-filters--authorship').boundingBox();
        assert.ok(authorRow.y >= row.y + row.height);
        const tops = await page.locator('button[data-publication-type]').evaluateAll(es => es.map(e => e.getBoundingClientRect().top));
        assert.ok(tops.every(top => top === tops[0]));
      }
      await page.locator('.publication-journals').scrollIntoViewIfNeeded();
      await page.screenshot({ path: `/private/tmp/publications-${width}.png` });
      assert.deepEqual(errors, []);
      console.log(`PASS ${width}px: ${total} structured cards, filters, badges, citations, typography and width`);
      await page.close();
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
