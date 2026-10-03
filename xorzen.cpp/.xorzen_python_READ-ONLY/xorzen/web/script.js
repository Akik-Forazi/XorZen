document.addEventListener('DOMContentLoaded', () => {
    // --- Smooth Scrolling ---
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // --- Typing Effect ---
    const subtitle = document.querySelector('.hero p');
    if (subtitle) {
        const text = subtitle.innerText;
        subtitle.innerText = '';
        let i = 0;
        const type = () => {
            if (i < text.length) {
                subtitle.innerText += text.charAt(i);
                i++;
                setTimeout(type, 30);
            }
        };
        type();
    }

    // --- Navbar Reveal on Scroll ---
    const nav = document.querySelector('nav');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            nav.style.background = 'rgba(5, 5, 7, 0.95)';
            nav.style.padding = '1rem 10%';
        } else {
            nav.style.background = 'rgba(5, 5, 7, 0.8)';
            nav.style.padding = '1.5rem 10%';
        }
    });

    // --- Dynamic Stats Counter (Simple) ---
    const counters = document.querySelectorAll('.stat-number');
    const speed = 200;

    const startCounters = () => {
        counters.forEach(counter => {
            const updateCount = () => {
                const target = +counter.getAttribute('data-target');
                const count = +counter.innerText;
                const inc = target / speed;

                if (count < target) {
                    counter.innerText = Math.ceil(count + inc);
                    setTimeout(updateCount, 1);
                } else {
                    counter.innerText = target;
                }
            };
            updateCount();
        });
    };

    // Intersection Observer for counters
    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            startCounters();
            observer.disconnect();
        }
    }, { threshold: 0.5 });

    if (counters.length > 0) {
        observer.observe(document.querySelector('.stats-container'));
    }
});
