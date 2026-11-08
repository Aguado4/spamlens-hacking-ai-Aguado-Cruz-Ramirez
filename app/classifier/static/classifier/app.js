// Progressive enhancement: without this script the form still posts and renders server-side.
(() => {
  const form = document.querySelector('[data-checker]');
  const field = form.querySelector('textarea');
  const submit = form.querySelector('[data-submit]');
  const count = form.querySelector('[data-count]');
  const max = Number(field.getAttribute('maxlength')) || 2000;

  const updateCount = () => {
    count.textContent = field.value.length;
    count.parentElement.classList.toggle('is-over', field.value.length > max);
  };
  field.addEventListener('input', updateCount);
  updateCount();

  // Submit with Ctrl/Cmd + Enter from the textarea.
  field.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) form.requestSubmit();
  });

  form.querySelectorAll('[data-sample]').forEach((chip) => {
    chip.addEventListener('click', () => {
      field.value = chip.dataset.sample;
      updateCount();
      form.requestSubmit();
    });
  });

  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!field.value.trim()) {
      field.focus();
      return;
    }
    submit.disabled = true;
    submit.classList.add('is-loading');
    submit.querySelector('.btn__text').textContent = 'Reading...';
    try {
      const res = await fetch(form.action, {
        method: 'POST',
        body: new FormData(form),
        headers: { 'X-Requested-With': 'fetch' },
      });
      if (!res.ok) throw new Error(res.statusText);
      const html = await res.text();
      document.getElementById('result').outerHTML = html;
      const result = document.getElementById('result');
      if (result.getBoundingClientRect().bottom > window.innerHeight) {
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    } catch {
      document.getElementById('result').innerHTML =
        '<p class="result__error" role="alert">Something went wrong reaching the model. Try again in a moment.</p>';
    } finally {
      submit.disabled = false;
      submit.classList.remove('is-loading');
      submit.querySelector('.btn__text').textContent = 'Check my message';
    }
  });

  // Scroll reveal.
  const io = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });
  document.querySelectorAll('.reveal').forEach((el) => io.observe(el));

  const topbar = document.querySelector('.topbar');
  const onScroll = () => topbar.classList.toggle('is-scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();
