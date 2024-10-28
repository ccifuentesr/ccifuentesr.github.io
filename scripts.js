document.addEventListener('DOMContentLoaded', function() {
    gsap.registerPlugin(ScrollTrigger);
    gsap.from(".logo img", {duration: 2, opacity: 0, y: -50, ease: "bounce.out"});
    gsap.from(".navbar ul li", {duration: 1, opacity: 0, y: 20, stagger: 0.2, delay: 0.5});

    gsap.from(".section__headline", {
        scrollTrigger: {
            trigger: ".section__headline",
            start: "top 80%", 
        },
        duration: 1,
        opacity: 0,
        y: 50
    });

    gsap.from(".section__content", {
        scrollTrigger: {
            trigger: ".section__content",
            start: "top 80%", 
        },
        duration: 1,
        opacity: 0,
        y: 50,
        delay: 0.3
    });
});

document.getElementById('menu-toggle').addEventListener('click', function() {
    this.classList.toggle('active');
    const navMenu = document.getElementById('nav-menu');
    if (this.classList.contains('active')) {
        navMenu.style.display = 'flex';
    } else {
        navMenu.style.display = 'none';
    }
});

  