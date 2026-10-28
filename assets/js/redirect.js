const destinations = { about: 'index.html', positions: 'cv.html#experience', questions: 'cv.html#experience', projects: 'projects.html', code: 'projects.html', 'selected-work': 'projects.html', publications: 'publications.html', contact: 'contact.html' };
location.replace(destinations[location.hash.slice(1)] || 'projects.html');
