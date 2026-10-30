'use strict';

const backToTop = document.querySelector('.scroll-to-top');
const syncScrollButton = () => backToTop.classList.toggle('visible', window.scrollY > 500);
window.addEventListener('scroll', syncScrollButton, { passive: true });
syncScrollButton();

const copyButton = document.querySelector('.copy-bibtex-btn');
const copyStatus = document.querySelector('.copy-status');
copyButton.addEventListener('click', async () => {
  const citation = document.querySelector('#bibtex-code code').textContent;
  try {
    await navigator.clipboard.writeText(citation);
    copyButton.textContent = 'Copied!';
    copyStatus.textContent = 'Citation copied to clipboard.';
    window.setTimeout(() => { copyButton.textContent = 'Copy citation'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex-code'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Use your browser’s Copy command, or download the .bib file.';
  }
});

const dialog = document.querySelector('#figure-dialog');
const enlargedFigure = document.querySelector('#enlarged-figure');
let figureTrigger;
document.querySelectorAll('[data-zoom]').forEach(link => {
  link.addEventListener('click', event => {
    if (!dialog.showModal || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    const original = link.querySelector('img');
    figureTrigger = link;
    enlargedFigure.src = link.href;
    enlargedFigure.alt = original.alt;
    enlargedFigure.classList.toggle('portrait', original.naturalHeight > original.naturalWidth * 0.85);
    document.querySelector('#figure-dialog-title').textContent = original.alt;
    dialog.showModal();
    document.body.classList.add('figure-open');
    dialog.querySelector('.zoom-scroll').scrollTo(0, 0);
  });
});
dialog.querySelector('.close-dialog').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) {
  const box = dialog.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom) dialog.close();
}});
dialog.addEventListener('close', () => {
  document.body.classList.remove('figure-open');
  figureTrigger?.focus({ preventScroll: true });
});

const rolloutViewers = [...document.querySelectorAll('[data-rollout-viewer]')];
if (rolloutViewers.length) {
  const carousel = document.querySelector('#rollout-viewer');
  const tabs = [...carousel.querySelectorAll('[role="tab"]')];
  let activeSlide = 0;
  const showSlide = (index, focusTab = false) => {
    activeSlide = (index + rolloutViewers.length) % rolloutViewers.length;
    rolloutViewers.forEach((viewer, i) => {
      viewer.hidden = i !== activeSlide;
      if (viewer.hidden) viewer.querySelector('video').pause();
      tabs[i].setAttribute('aria-selected', String(i === activeSlide));
      tabs[i].tabIndex = i === activeSlide ? 0 : -1;
    });
    if (focusTab) tabs[activeSlide].focus({ preventScroll: true });
  };
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => showSlide(index));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = activeSlide + 1;
      else if (event.key === 'ArrowLeft') next = activeSlide - 1;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = rolloutViewers.length - 1;
      else return;
      event.preventDefault();
      showSlide(next, true);
    });
  });
  carousel.querySelector('[data-rollout-prev]').addEventListener('click', () => showSlide(activeSlide - 1));
  carousel.querySelector('[data-rollout-next]').addEventListener('click', () => showSlide(activeSlide + 1));
  carousel.querySelector('.rollout-carousel-controls').hidden = false;
  carousel.classList.add('is-carousel');
  showSlide(0);

  const base = 'static/videos/molmo/';
  const manifest = fetch(`${base}selected_examples.json`).then(response => {
    if (!response.ok) throw new Error('Video manifest unavailable');
    return response.json();
  });
  rolloutViewers.forEach(viewer => {
    const video = viewer.querySelector('video');
    const buttons = [...viewer.querySelectorAll('[data-rollout-task]')];
    let selectedTask = Number(viewer.dataset.defaultTask);
    let examples = [];

    const showRollout = () => {
      const example = examples.find(item => item.task_id === selectedTask);
      if (!example?.combined_file) return;
      const source = `${base}${example.combined_file}`;
      video.pause();
      video.src = source;
      video.poster = source.replace(/\.mp4$/, '.jpg');
      video.setAttribute('aria-label', `BoN ${example.object.toLowerCase()} rollouts with 64 and 144 training tasks, synchronized wrist and exterior cameras`);
      video.load();
      buttons.forEach(button => button.setAttribute('aria-pressed', String(Number(button.dataset.rolloutTask) === selectedTask)));
      video.querySelector('a').href = source;
    };

    buttons.forEach(button => button.addEventListener('click', () => {
      if (selectedTask === Number(button.dataset.rolloutTask)) return;
      selectedTask = Number(button.dataset.rolloutTask);
      showRollout();
    }));
    video.addEventListener('play', () => {
      rolloutViewers.forEach(other => {
        if (other !== viewer) other.querySelector('video').pause();
      });
    });
    manifest.then(data => {
      if (!Array.isArray(data.examples)) throw new Error('No video examples available');
      examples = data.examples;
      buttons.forEach(button => {
        button.disabled = !examples.some(item => item.task_id === Number(button.dataset.rolloutTask) && item.combined_file);
      });
      showRollout();
    }).catch(() => {
      buttons.forEach(button => { button.disabled = true; });
    });
  });
}
