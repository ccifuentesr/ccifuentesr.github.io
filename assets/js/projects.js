// ===== THESIS CHAT FUNCTIONS =====
        function showProgress() {
            const bar = document.getElementById("progressBar");
            let width = 0;
            return setInterval(() => {
                if (width < 90) {
                    width += 1;
                    bar.style.width = width + "%";
                }
            }, 50);
        }

        function addMessage(content, type) {
            const messagesEl = document.getElementById("chatMessages");
            const messageDiv = document.createElement('div');
            messageDiv.style.cssText = 'padding: 0.8em; background: ' + (type === 'user' ? 'var(--text-color)' : 'var(--hover-bg)') + '; color: ' + (type === 'user' ? 'var(--bg-color)' : 'var(--text-color)') + '; border-radius: 6px; margin-bottom: 0.8em;';
            messageDiv.innerHTML = '<p style="margin: 0; font-size: 0.9em;">' + content + '</p>';
            messagesEl.appendChild(messageDiv);
            messagesEl.scrollTop = messagesEl.scrollHeight;
        }

        async function askThesis() {
            const question = document.getElementById("question").value.trim();
            if (!question) return;

            addMessage(question, 'user');

            const btn = document.getElementById("askBtn");
            const container = document.getElementById("progressContainer");

            document.getElementById("question").value = '';

            btn.disabled = true;
            btn.style.opacity = 0.6;
            container.style.display = "block";

            const interval = showProgress();

            try {
                const res = await fetch("https://ccifuentesr-github-io.onrender.com/ask", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ question })
                });

                const data = await res.json();
                clearInterval(interval);

                const bar = document.getElementById("progressBar");
                bar.style.width = "100%";

                container.style.transition = "opacity 0.3s ease";
                container.style.opacity = 0;
                setTimeout(() => {
                    container.style.display = "none";
                    container.style.opacity = 1;
                    bar.style.width = "0%";
                }, 300);

                const rawAnswer = data.answer || "No answer.";
                let formatted = rawAnswer
                    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
                    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
                    .replace(/^# (.*$)/gim, "<h1>$1</h1>")
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                    .replace(/\*(.*?)\*/g, "<em>$1</em>")
                    .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>')
                    .replace(/^\s*[-+*]\s+(.*)/gim, "<li>$1</li>")
                    .replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>")
                    .replace(/\n/g, "<br>");

                if (data.chunks && data.chunks.length > 0) {
                    formatted += "<br><br><strong>Sources</strong><ul>";
                    data.chunks.forEach(c => {
                        formatted += `<li>${c}</li>`;
                    });
                    formatted += "</ul>";
                }

                addMessage(formatted, 'bot');

            } catch (err) {
                clearInterval(interval);
                container.style.display = "none";
                addMessage("Uh-oh, I'm lost for words...", 'bot');
            } finally {
                btn.disabled = false;
                btn.style.opacity = 1;
            }
        }

        document.getElementById("question").addEventListener("keypress", function(e) {
            if (e.key === "Enter") {
                askThesis();
            }
        });

        // ===== CATALOGUE SEARCH FUNCTIONS =====
        function updateSearchPlaceholder() {
            const catalog = document.getElementById('catalog-select').value;
            const searchInput = document.getElementById('unified-search');
            
            if (!searchInput) return;
            
            const karmnCatalogues = ['cifuentes25', 'cifuentes20', 'schweitzer19', 'cortes-contreras24'];
            
            if (karmnCatalogues.includes(catalog)) {
                searchInput.placeholder = 'Object name or Karmn identifier (e.g., J00026+383)...';
            } else {
                searchInput.placeholder = 'Object name...';
            }
        }

        async function unifiedSearch() {
            const catalog = document.getElementById('catalog-select').value;
            const searchTerm = document.getElementById('unified-search').value.trim();
            const resultElement = document.getElementById('unified-result');

            if (!catalog) {
                resultElement.style.display = 'block';
                resultElement.style.background = 'var(--hover-bg)';
                resultElement.style.border = '1px solid var(--border-color)';
                resultElement.innerHTML = '<p style="color: var(--text-secondary); margin:0;">Please select a catalogue first.</p>';
                return;
            }

            if (!searchTerm) {
                resultElement.style.display = 'block';
                resultElement.style.background = 'var(--hover-bg)';
                resultElement.style.border = '1px solid var(--border-color)';
                resultElement.innerHTML = '<p style="color: var(--text-secondary); margin:0;">Please enter an object name to search.</p>';
                return;
            }

            let vizierSource;
            if (catalog === 'cifuentes25') {
                vizierSource = 'J/A+A/693/A228';
            } else if (catalog === 'cifuentes20') {
                vizierSource = 'J/A+A/642/A115';
            } else if (catalog === 'schweitzer19') {
                vizierSource = 'J/A+A/625/A68';
            } else if (catalog === 'martinez-rodriguez19') {
                vizierSource = 'J/ApJ/887/261';
            } else if (catalog === 'gonzalez-payo24') {
                vizierSource = 'J/A+A/689/A302';
            } else if (catalog === 'cortes-contreras24') {
                vizierSource = 'J/A+A/692/A206';
            }

            resultElement.style.display = 'block';
            resultElement.style.background = 'var(--bg-color)';
            resultElement.style.border = '1px solid var(--border-color)';
            resultElement.innerHTML = '<p style="color: var(--text-secondary); font-style:italic; margin:0;">Searching...</p>';

            try {
                const karmnCatalogues = ['cifuentes25', 'cifuentes20', 'schweitzer19', 'cortes-contreras24'];
                
                let tsvUrl;
                const isKarmnId = searchTerm.match(/^J\d{5}[+-]\d{3}$/i);
                
                if (isKarmnId && karmnCatalogues.includes(catalog)) {
                    tsvUrl = `https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=${encodeURIComponent(vizierSource)}&Karmn=${encodeURIComponent(searchTerm)}&-out.max=50&-out.all`;
                } else {
                    tsvUrl = `https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=${encodeURIComponent(vizierSource)}&-c=${encodeURIComponent(searchTerm)}&-out.max=1&-out.all`;
                }
                
                const resp = await fetch(tsvUrl);

                if (!resp.ok) {
                    resultElement.style.display = 'block';
                    const vizierUrl = `https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=${encodeURIComponent(vizierSource)}&-c=${encodeURIComponent(searchTerm)}&-out.max=1&-out.form=HTML`;
                    resultElement.innerHTML = `<iframe src="${vizierUrl}" style="width: 100%; height: 600px; border: none;" title="VizieR Catalogue Results"></iframe>`;
                    return;
                }

                const text = await resp.text();
                
                const lines = text.split('\n')
                    .map(l => l.trim())
                    .filter(l => l && !l.startsWith('#') && !/^[-=\s\t]+$/.test(l));
                
                if (lines.length < 3) {
                    resultElement.style.background = 'var(--hover-bg)';
                    resultElement.style.border = '1px solid var(--border-color)';
                    const catalogueUrl = `https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=${vizierSource}`;
                    resultElement.innerHTML = `<p style="color: var(--text-secondary); margin:0;">Object not found in this <a href="${catalogueUrl}" target="_blank" style="color: #0084ff; font-weight: 600; text-decoration: underline;">catalogue</a>.</p>`;
                    return;
                }

                const headers = lines[0].split('\t').map(h => h.split(' [')[0].trim());
                const units = lines[1].split('\t').map(u => u.trim());
                const data = lines[2].split('\t');

                const readmeUrl = `https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/${vizierSource}?format=html&tex=true`;
                const catalogueUrl = `https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=${vizierSource}`;

                let html = `<div style="margin-bottom: 1em; padding: 0.8em; background: var(--hover-bg); border-left: 3px solid var(--text-color); font-size: 0.9em; color: var(--text-color);">
                    For detailed information about this <a href="${catalogueUrl}" target="_blank" style="color: #0084ff; font-weight: 600; text-decoration: underline;">catalogue</a>, see the <a href="${readmeUrl}" target="_blank" style="color: #0084ff; font-weight: 600; text-decoration: none;">ReadMe file</a>.
                </div>`;
                
                html += `<div style="font-size: 0.9em; color: var(--text-color);">`;
                
                headers.forEach((header, index) => {
                    const value = data[index] ? data[index].trim() : '';
                    if (value && value !== '---') {
                        let display = `<strong style="color: var(--text-color);">${header}:</strong> ${value}`;
                        const unit = units[index] || '';
                        if (unit && unit !== '---') {
                            display += ` <span style="color: var(--text-secondary);">(${unit})</span>`;
                        }
                        html += `<div style="margin-bottom: 0.5em; color: var(--text-color);">${display}</div>`;
                    }
                });
                html += '</div>';

                resultElement.style.background = 'var(--bg-color)';
                resultElement.style.border = '1px solid var(--border-color)';
                resultElement.innerHTML = html;

            } catch (err) {
                console.error('Catalogue search error:', err);
                resultElement.style.display = 'block';
                const vizierUrl = `https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=${encodeURIComponent(vizierSource)}&-c=${encodeURIComponent(searchTerm)}&-out.max=1&-out.form=HTML`;
                resultElement.innerHTML = `<iframe src="${vizierUrl}" style="width: 100%; height: 600px; border: none;" title="VizieR Catalogue Results"></iframe>`;
            }
        }

        document.getElementById('unified-search').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                unifiedSearch();
            }
        });
