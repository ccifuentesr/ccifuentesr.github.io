const audio = document.getElementById('global-audio');
const player = document.getElementById('global-player');
const playButton = player.querySelector('.play-btn');
const seek = document.getElementById('audio-seek');
const timeLabel = seconds => Number.isFinite(seconds) ? `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}` : '0:00';
function togglePlayGlobal() {
  if (!audio.src) return;
  if (audio.paused) audio.play().catch(() => {}); else audio.pause();
}
function loadTrack(src, el) {
  closeLyrics();
  player.hidden = false;
  if (audio.src !== new URL(src, location.href).href) { audio.src = src; seek.value = 0; }
  document.querySelectorAll('.music-track').forEach(row => row.classList.remove('playing'));
  el.classList.add('playing');
  document.getElementById('now-playing').textContent = (el.querySelector('.track-name') || el.querySelector('span') || el).textContent.trim();
  audio.play().catch(() => {});
}
function syncPlayback() {
  playButton.textContent = audio.paused ? '▶' : 'Ⅱ';
  playButton.setAttribute('aria-label', audio.paused ? 'Play' : 'Pause');
}
['play','pause','ended'].forEach(event => audio.addEventListener(event, syncPlayback));
audio.addEventListener('ended', () => document.querySelectorAll('.playing').forEach(el => el.classList.remove('playing')));
function syncTime() {
  seek.value = Number.isFinite(audio.duration) && audio.duration > 0 ? audio.currentTime / audio.duration * 100 : 0;
  document.getElementById('global-current-time').textContent = timeLabel(audio.currentTime);
  document.getElementById('global-duration').textContent = timeLabel(audio.duration);
}
['timeupdate','loadedmetadata','emptied'].forEach(event => audio.addEventListener(event, syncTime));
seek.addEventListener('input', () => { if (Number.isFinite(audio.duration)) audio.currentTime = Number(seek.value) / 100 * audio.duration; });
document.querySelectorAll('.music-track').forEach(row => row.addEventListener('keydown', event => {
  if (event.target === row && (event.key === 'Enter' || event.key === ' ')) { event.preventDefault(); row.click(); }
}));
        const lyricsDatabase = {
            triangles: {
                title: "TRIANGLES (2018)",
                lyrics: `If the triangles made a god
If the triangles made a god
They would give him three sides`
            },
            dormant: {
                title: "DORMANT (2019)",
                lyrics: `Again you fail to see
It's no one but you
The powers that be`
            },
            haunt: {
                title: "HAUNT (2019)",
                lyrics: `There is no plan
There is no purpose
Zweck sein selbst ist jegliches Tier
(Each animal is an end in itself)`
            },
            bones: {
                title: "BONES (2019)",
                lyrics: `[These lyrics are lost forever]`
            },
            puma: {
                title: "PUMA (2021)",
                lyrics: `Eumetazoa
Deuterostomia
Chordata
Vertebrata
Gnathostomata
Tetrapoda
Mammalia
Theria
Placentalia
Laurasiatheria
Carnivora
Feliformia
Felidae
Felinae
Puma
Puma concolor`
            },
            pig: {
                title: "PIG (2024)",
                lyrics: `(Instrumental)`
            }
        };

        function showLyrics(songId) {
            const song = lyricsDatabase[songId];
            if (!song) {
                alert('Letras no disponibles para esta canción.');
                return;
            }
            
            closeLyrics();
            // Native modal keeps lyrics separate from page content.
            const modal = document.createElement('dialog');
            modal.className = 'lyrics-modal';
            modal.setAttribute('aria-labelledby', 'lyrics-title');
            let trivia = '';
            if (songId === 'triangles') {
                trivia = `<div class="lyrics-trivia" style="margin-top:1.5em; color:var(--text-color); font-size:0.98em;">
                <span style="font-weight: bold; color: var(--text-color);">Trivia:</span><br>
                <ul style="margin:0.15em 0 0 1.1em; padding:0; list-style-type:disc;">
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">The intro riff was repeated at least 100 times.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">With the exception of some fills, the song was improvised in one take.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Guitar recorded with Windows Sound Recorder (.wav converted to .mp3).</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Left index finger bruised after repeatedly hitting a prominent edge of the Showmaster neck.</li>
                </ul>
                </div>`;
            } else if (songId === 'bones') {
                trivia = `<div class="lyrics-trivia" style="margin-top:1.5em; color:var(--text-color); font-size:0.98em;">
                <span style="font-weight: bold; color: var(--text-color);">Trivia:</span><br>
                <ul style="margin:0.15em 0 0 1.1em; padding:0; list-style-type:disc;">
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">The intro riff was repeated approximately 500 times.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Around 1,500 audio fragments were recorded but not used in the final mix.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Vocals include Rottweiler and German Shepherd samples.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">The main microphone failed and was replaced with an iPhone.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Recording, production, and mixing were completed in about 72 hours.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Released on Good Friday.</li>
                </ul>
                </div>`;
            } else if (songId === 'dormant') {
                trivia = `<div class="lyrics-trivia" style="margin-top:1.5em; color:var(--text-color); font-size:0.98em;">
                <span style="font-weight: bold; color: var(--text-color);">Trivia:</span><br>
                <ul style="margin:0.15em 0 0 1.1em; padding:0; list-style-type:disc;">
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">The intro riff was repeated at least one hundred times.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Guitar played while barefoot like Steven Wilson.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Some sounds achieved with scissors.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Officially released on Maundy Thursday.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">A broken PC was used for recording.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Recording, production, and mixing took less than 48 hours.</li>
                </ul>
                </div>`;
            } else if (songId === 'haunt') {
                trivia = `<div class="lyrics-trivia" style="margin-top:1.5em; color:var(--text-color); font-size:0.98em;">
                <span style="font-weight: bold; color: var(--text-color);">Trivia:</span><br>
                <ul style="margin:0.15em 0 0 1.1em; padding:0; list-style-type:disc;">
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Recorded sitting on a wooden bench, eating pickles.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">Features an undefined jungle animal, probably a lemur.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">The song that has caused the most fingertip pain.</li>
                    <li style="margin:0; padding:0; line-height:0.95; display:list-item;">First song conceived in Guitar Pro (originally called 'Melting Sun').</li>
                </ul>
                </div>`;
            }
            modal.innerHTML = `
                <div class="lyrics-content">
                    <div class="lyrics-header">
                        <h3 id="lyrics-title">${song.title}</h3>
                        <button type="button" class="close-lyrics" onclick="closeLyrics()" aria-label="Close lyrics">×</button>
                    </div>
                    <div class="lyrics-text">
                        <span style="font-weight: bold; color: var(--text-color);">Human lyrics:</span><br>
                        <span style="font-style: italic;">${song.lyrics.replace(/\n/g, '<br>')}</span>
                        ${trivia}
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            modal.showModal();
            modal.addEventListener("close", () => modal.remove());
            
            modal.addEventListener('click', (e) => {
                if (e.target === modal) closeLyrics();
            });
        }
        
        function closeLyrics() {
            const modal = document.querySelector('.lyrics-modal');
            if (modal) { modal.close(); modal.remove(); }
        }
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLyrics();
        });
    
window.addEventListener('hashchange', closeLyrics);
