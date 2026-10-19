const contactForm = document.getElementById('contact-form');

if (contactForm) {
    const status = document.getElementById('contact-status');
    const submitButton = contactForm.querySelector('button[type="submit"]');

    contactForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        status.textContent = '';
        delete status.dataset.state;
        submitButton.disabled = true;
        submitButton.textContent = 'Sending…';

        try {
            const response = await fetch(contactForm.action, {
                method: 'POST',
                body: new FormData(contactForm),
                headers: { Accept: 'application/json' }
            });

            if (!response.ok) {
                throw new Error('The form service could not accept the message.');
            }

            contactForm.reset();
            status.dataset.state = 'success';
            status.textContent = 'Your message was sent. Thank you!';
        } catch (error) {
            status.dataset.state = 'error';
            status.textContent = 'Your message could not be sent. Please try again later.';
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = 'Send message';
        }
    });
}
