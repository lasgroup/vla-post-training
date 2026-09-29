# Research project website

Static GitHub Pages site for **Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement**, in the `gh-pages/` directory on the `gh-pages` branch.

## Preview

Run `python3 -m http.server 8000 --directory gh-pages` from the repository root and open <http://localhost:8000>. No build step or package installation is required. Styles, scripts, figures, and the manuscript are served locally; there are no third-party runtime dependencies.

## Publish

The workflow in [`.github/workflows/pages.yml`](../.github/workflows/pages.yml) publishes the contents of `gh-pages/` as the website root using GitHub Actions. It runs when website files or the workflow change on the `gh-pages` branch, and supports manual dispatch on that branch. It checks out only `gh-pages/`, then uploads and deploys those static files directly. It does not build the repository or install project dependencies.

One-time setup:

1. In the repository's **Settings → Pages → Build and deployment**, set **Source** to **GitHub Actions**.
2. If the `github-pages` environment restricts deployment branches, allow `gh-pages` under **Settings → Environments → github-pages**.
3. Commit the website directory and `.github/workflows/pages.yml`, then push the `gh-pages` branch. Follow the **Deploy research website** run in the Actions tab.

The deployed site will be <https://lasgroup.github.io/vla-post-training/>. The folder name does not add another `/gh-pages/` to the public URL. Deployment uses the built-in `GITHUB_TOKEN`; no personal access token is needed. The workflow and website must be pushed before deployment can run.

See GitHub's [custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages) for repository setup and deployment permissions.

## Content and provenance

- Template: `/Users/yardas/vla-post-training-website/`, preserving its title/authors/resource links, teaser, abstract, motivation, evaluation, highlights, citation, and attribution structure. Bulma is copied from the supplied template; page styling and JavaScript are adapted for this paper.
- Source: `_ICLR__27__Multi_task_policy_learning_SwissAI (1).zip`, supplied from Downloads. Figure variants are the ones referenced by the active paper source, not older alternatives.
- `static/paper/generalist-policy-improvement.pdf` is compiled from that ZIP with `latexmk -pdf main_ICLR.tex`; it retains the source manuscript's anonymous submission formatting. The website author list and affiliations were supplied separately by the user. Department and alternate-name entries are consolidated into eight institutions (ETH Zurich, MPI for Intelligent Systems, TUM, CMU, University of Warsaw, UC Berkeley, Microsoft, and UT Austin), with author numbers remapped consistently. The generic Max-Planck entry was matched to MPI for Intelligent Systems using [Leander Diaz-Bone’s profile](https://leanderdiazbone.github.io/). No affiliations were inferred for authors without a number.
- All numerical statements describe the manuscript's simulation experiments. The page labels the manuscript as a submission, not an accepted publication.
- Source figure mapping: `unseen-objects` → `hook_variant_3`; `scaling` → `main_figure_variant_5`; `generalization` → `many_categories_generalization_variant_7`; `recipe-ablations` → `all_ablations`; `policy-extraction` → `actor_critic_ablation_variant_3`. All plot images, enlarged views, and figure downloads use SVG converted directly from the source PDFs with Poppler (`pdftocairo -svg input.pdf output.svg`). The SVGs contain vector paths, including outlined labels, with no embedded raster images or external font dependencies. PDF originals are retained as source assets; the full manuscript is still available as a PDF. The PNG social-preview image is only used for link previews.
- Task images and labels follow `figs/molmo_visualization_small.tex`.

Paths below are relative to `gh-pages/`. Edit `index.html` for text/authors/citation, `static/css/index.css` for styling, and `static/js/index.js` for figure zoom and clipboard controls. Keep the citation in `static/paper/citation.bib` consistent with the displayed BibTeX.

## Attribution

Website template adaptations are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), preserving the supplied template's attribution to the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) and [Nerfies](https://nerfies.github.io/). The template license does not relicense the research manuscript, figures, or repository code; the existing repository `LICENSE` is unchanged.

## Verification

Visually reviewed browser screenshots at 1440px (desktop), 768px (tablet), 390px and 360px (phones). Checked image loading, horizontal overflow, local resource responses, section anchors, all 24 author entries, affiliation expansion, figure zoom and Escape dismissal, and citation clipboard/download consistency. No missing local resources or JavaScript errors were found. The site uses relative asset paths and was tested under the `/vla-post-training/` project path.
