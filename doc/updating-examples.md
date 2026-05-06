# Updating Sphinx-Gallery Examples

## How it works

Sphinx-Gallery stores an MD5 checksum of each example script next to its
generated outputs (under `doc/auto_examples/`).  At build time it compares
the checksum of the source script with the cached one:

- **Match** → skip execution, reuse existing images and RST.
- **Mismatch / missing** → re-execute the script, regenerate images and RST, write new MD5.

Because `doc/auto_examples/` is committed to the repository (with images
tracked via Git LFS), Read the Docs finds matching MD5s on every build and
never needs to re-execute the scripts.  This keeps RTD builds fast and
within the 15-minute time limit.

## Workflow when editing an example

1. **Edit the source script**

   ```
   examples/fake_data/plot_stacking.py   # or plot_simstack.py, etc.
   ```

2. **Delete the cached MD5** for that script so Sphinx-Gallery knows it must re-run it

   ```bash
   rm doc/auto_examples/fake_data/plot_stacking.py.md5
   ```

3. **Rebuild the docs locally** — Sphinx-Gallery re-executes the script,
   regenerates the images, and writes a fresh MD5

   ```bash
   source ~/bin/miniconda3/etc/profile.d/conda.sh && conda activate nikamap
   sphinx-build -b html doc doc/_build/html
   ```

4. **Commit everything** — source script, updated RST, new images (LFS), new MD5

   ```bash
   git add examples/fake_data/plot_stacking.py
   git add doc/auto_examples/fake_data/
   git commit -m "docs: update stacking example"
   git push
   ```

## Important rules

- **Never call `plt.close('all')` in an example script.**  Sphinx-Gallery
  captures figures that are still open at the end of each code block.
  Closing figures before the capture step produces pages with no images.

- **Always commit `doc/auto_examples/`** after a local rebuild.  If you push
  only the source script without updating the cache, RTD will detect a MD5
  mismatch and attempt to re-execute — which will fail or time out.

- **Binary files are in Git LFS** (`.png`, `.zip`, `.ipynb` under
  `doc/auto_examples/`).  Make sure `git lfs` is installed locally before
  committing (`git lfs install` once per machine).
