# Research project website

Static GitHub Pages site for **Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement**, in the `gh-pages/` directory on the `gh-pages` branch.

## Preview

Run `python3 -m http.server 8000 --directory gh-pages` from the repository root and open <http://localhost:8000>. No build step or package installation is required. Styles, scripts, figures, and the manuscript are served locally; there are no third-party runtime dependencies.

## Publish

Publish the contents of `gh-pages/` as the website root, for example by uploading that directory as a GitHub Pages deployment artifact. The repository root is no longer the website root. The site is designed for <https://lasgroup.github.io/vla-post-training/>; publishing the directory contents at that URL preserves the canonical and social-preview URLs. `.nojekyll` is included for static serving.

## Content and provenance

- Template: `/Users/yardas/vla-post-training-website/`, preserving its title/authors/resource links, teaser, abstract, motivation, method, evaluation, highlights, citation, and attribution structure. Bulma is copied from the supplied template; page styling and JavaScript are adapted for this paper.
- Source: `_ICLR__27__Multi_task_policy_learning_SwissAI (1).zip`, supplied from Downloads. Figure variants are the ones referenced by the active paper source, not older alternatives.
- `static/paper/generalist-policy-improvement.pdf` is compiled from that ZIP with `latexmk -pdf main_ICLR.tex`; it retains the source manuscript's anonymous submission formatting. The website author list and affiliation numbering were supplied separately by the user. No affiliations were inferred for authors without a number.
- All numerical statements describe the manuscript's simulation experiments. The page labels the manuscript as a submission, not an accepted publication.
- Source figure mapping: `unseen-objects` → `hook_variant_3`; `scaling` → `main_figure_variant_5`; `generalization` → `many_categories_generalization_variant_7`; `recipe-ablations` → `all_ablations`; `policy-extraction` → `actor_critic_ablation_variant_3`. PNGs are rendered at 300 dpi; PDF originals remain downloadable.
- Task images and labels follow `figs/molmo_visualization_small.tex`.

Paths below are relative to `gh-pages/`. Edit `index.html` for text/authors/citation, `static/css/index.css` for styling, and `static/js/index.js` for figure zoom and clipboard controls. Keep the citation in `static/paper/citation.bib` consistent with the displayed BibTeX.

## Attribution

Website template adaptations are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), preserving the supplied template's attribution to the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) and [Nerfies](https://nerfies.github.io/). The template license does not relicense the research manuscript, figures, or repository code; the existing repository `LICENSE` is unchanged.

## Verification

Visually reviewed browser screenshots at 1440px (desktop), 768px (tablet), 390px and 360px (phones). Checked image loading, horizontal overflow, local resource responses, section anchors, all 24 author entries, affiliation expansion, figure zoom and Escape dismissal, and citation clipboard/download consistency. No missing local resources or JavaScript errors were found. The site uses relative asset paths and was tested under the `/vla-post-training/` project path.
