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

- Template: `/Users/yardas/vla-post-training-website/`, preserving its title/authors/resource links, teaser, summary (now TL;DR), motivation, evaluation, highlights, citation, and attribution structure. Bulma is copied from the supplied template; page styling and JavaScript are adapted for this paper.
- Source: `_ICLR__27__Multi_task_policy_learning_SwissAI (1).zip`, supplied from Downloads. The teaser and scaling plots were subsequently replaced with the supplied `hook_variant_4.pdf` and `main_figure_variant_8.pdf`, extending the displayed training-task range to 144. Other figure variants follow the active paper source.
- `static/paper/generalist-policy-improvement.pdf` is compiled from that ZIP with `latexmk -pdf main_ICLR.tex`; it retains the source manuscript's anonymous submission formatting. The website author list and affiliations were supplied separately by the user. Department and alternate-name entries are consolidated into eight institutions (ETH Zurich, MPI for Intelligent Systems, TUM, CMU, University of Warsaw, UC Berkeley, Microsoft, and UT Austin), with author numbers remapped consistently. The generic Max-Planck entry was matched to MPI for Intelligent Systems using [Leander Diaz-Bone’s profile](https://leanderdiazbone.github.io/). No affiliations were inferred for authors without a number.
- All numerical statements describe the manuscript's simulation experiments. The page labels the manuscript as a submission, not an accepted publication.
- Source figure mapping: `unseen-objects` → `hook_variant_4`; `scaling` → `main_figure_variant_8`; `generalization` → `many_categories_generalization_variant_7`; `recipe-ablations` → `all_ablations`; `policy-extraction` → `actor_critic_ablation_variant_3`. All plot images, enlarged views, and figure downloads use SVG converted directly from the source PDFs. The teaser and scaling SVGs use PyMuPDF (`page.get_svg_image(text_as_path=True)`); the remaining SVGs use Poppler (`pdftocairo -svg input.pdf output.svg`). The SVGs contain vector paths, including outlined labels, with no embedded raster images or external font dependencies. PDF originals are retained as source assets; the full manuscript is still available as a PDF. The PNG social-preview image is only used for link previews.
- Task images and labels follow `figs/molmo_visualization_small.tex`.
- The per-object `shape-transfer` figure comes directly from the supplied
  `generalization_per_task_variant_2.pdf`; its SVG was exported with PyMuPDF and
  outlined text, and the original PDF is retained.

Paths below are relative to `gh-pages/`. Edit `index.html` for text/authors/citation, `static/css/index.css` for styling, and `static/js/index.js` for figure zoom and clipboard controls. Keep the citation in `static/paper/citation.bib` consistent with the displayed BibTeX.

## BoN rollout comparisons

The video viewer uses actual MolmoSpaces checkpoint rollouts, with exterior and
wrist views in `static/videos/molmo/`. The page has a two-slide carousel: improvement on
atomizer/wine bottle, and little additional improvement on fork/knife/remote control.
Each video shows wrist above exterior and 64 tasks beside 144 tasks on one
synchronized playback timeline. Both panels start from the same simulator
state and play at native speed; the shorter rollout holds its final frame.
The exports use H.264, `yuv420p`, and fast-start metadata. Only the selected video
loads when played. Each slide has its own object selector; starting one video
pauses the other. The original separate-camera exports remain available.

The 64-task checkpoints use step 499999 and the 144-task checkpoints use step
750000. The combined videos display only training-task counts;
checkpoint steps, outcomes, seeds, and initial-state details remain in the manifest. `selected_examples.json` records the
training seed, episode, outcomes, selection rule, and all audited paired outcomes.
Wine bottle can be selected manually for a clearly visible difference; control
examples show both models succeeding. Fork uses training seed 1, episode 2;
remote control uses training seed 0, episode 3 (episode numbers are one-based).
These were selected from the existing recordings without additional sampling.
These clips illustrate individual behaviors;
the supplied figure is the separate aggregate result.

Recording, audit, and export scripts are in the main checkout under
`scripts/molmo_videos/`. After the initial atomizer trials all succeeded with
64 tasks, a second fixed batch extends seed 1 to 12 matched initial states,
keeping both checkpoints unchanged. Every attempt is retained and the manifest
discloses this additional sampling. The complete set contains 108 episodes
(216 raw camera videos), with five selected pairs exported in both camera views
and five combined-camera videos. Run `combine_camera_views.py` after composing
the individual camera views to regenerate the combined videos and manifest.

## Attribution

Website template adaptations are licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), preserving the supplied template's attribution to the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) and [Nerfies](https://nerfies.github.io/). The template license does not relicense the research manuscript, figures, or repository code; the existing repository `LICENSE` is unchanged.

## Verification

Visually reviewed browser screenshots at 1440px (desktop), 768px (tablet), 390px and 360px (phones). Checked image loading, horizontal overflow, local resource responses, section anchors, all 24 author entries, affiliation expansion, figure zoom and Escape dismissal, and citation clipboard/download consistency. No missing local resources or JavaScript errors were found. The site uses relative asset paths and was tested under the `/vla-post-training/` project path.

The rollout viewer was checked in Chrome at desktop and 390px phone widths,
including object and camera selection. All 108 recorded episodes passed checks
for completeness, checkpoint provenance, frame counts, native playback rate,
and browser-compatible encoding. All 54 paired simulator initial states and
initial camera images match exactly. All three Slurm jobs completed with exit 0.

The comparison blocks are capped at 860px wide, with video height adapting to
the viewport to fit a laptop screen. Repeated object headings and visible
provenance captions have been removed.

The four added-category cards use 1024px renders of the actual MolmoSpaces
target meshes: salt shaker (task 124), soap dispenser (12), mug (160), and fruit
(241). `scripts/molmo_videos/render_object_closeups.py` isolates target visual
geometry, renders a close camera at native output resolution, and obtains
transparency from the simulator segmentation pass. No image upscaling is used.
The visible rollout disclosure paragraph was removed at the author's request;
the full selection manifest remains with the video assets.

The comparison carousel supports direct tabs, previous/next buttons, and
Left/Right/Home/End keys on the tabs. Switching slides pauses the hidden video
and retains the chosen object and playback position. It does not auto-advance.
