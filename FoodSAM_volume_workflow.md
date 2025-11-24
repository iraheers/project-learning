# Food Volume Estimation Workflow

This note captures the end‑to‑end process we now use to recover food volumes for the Nutrition50 dishes. It is written so you can lift paragraphs directly into a report or slide deck.

---

## Artifacts We Start With

For every dish `dish_<id>` produced by FoodSAM we have the following on disk:

- `nutrition50_foodSAM_outputs/dish_<id>/pred_vis.png` – segmentation overlay used as a quick visual reference.
- `nutrition50_foodSAM_outputs/dish_<id>/sam_mask/*.png` – binary masks per segment (foreground + background).
- `nutrition50_foodSAM_outputs/dish_<id>/sam_mask_label/semantic_masks_category.txt` – mapping of mask id ➝ food label.
- `nutrition50_foodSAM_outputs/dish_<id>/sam_metadata.csv` – metadata (mask area, bounding box, confidence).
- `zoedepth_nutrition50_outputs/dish_<id>/enhance_vis_zoedepth.npy` – ZoeDepth depth map (metric depths in metres).

The processing notebooks live under `Week6/` and results are written to `Week6/nutrition50_volume/`.

---

## Volume Notebook (`Week6/FoodSAM_volume_analysis.ipynb`)

The notebook converts the segmentation + depth data into per‑food metric volumes. The major stages are:

1. **Data Load**  
   - Read FoodSAM metadata, per‑mask PNGs, and the ZoeDepth array.  
   - Build a `dish_id → category` lookup so background masks can be filtered out quickly.

2. **Auto Camera Calibration**  
   - We do not know the camera intrinsics, so we sweep plausible horizontal FOV values (45°–75°).  
   - For each candidate FOV we project background pixels into 3D, fit a support plane, and measure flatness.  
   - The FOV with the lowest plane RMSE is chosen and converted to pinhole intrinsics `(fx, fy, cx, cy)`.

3. **Support Plane Fit Per Mask**  
   - For each non‑background mask we dilate the mask to form a “support ring” in the surrounding background.  
   - The ring depths are back‑projected to 3D and a least‑squares plane is fit.  
   - Plane quality metrics (support pixel count, RMSE) are tracked so we can flag unreliable fits.

4. **Height Field & Volume Integration**  
   - Mask pixels are re‑projected into 3D using the calibrated intrinsics.  
   - Each point’s signed distance to the support plane gives a height above the plate.  
   - Only positive heights are kept. Per‑pixel area (`z² / (fx·fy)`) is used to convert heights into volume contributions.  
   - Summing height × pixel area yields the per‑mask volume in cubic metres → converted to millilitres.

5. **Aggregation & Export**  
   - Per‑mask results (volume, mean/max height, support diagnostics) are stored in `volumes_per_mask.csv`.  
   - Per‑category totals are accumulated in `volumes_per_category.csv`.  
   - A dish summary JSON records the selected FOV, intrinsics, support stats, and an interpreted “scale note.”  
   - All outputs for a dish live at `Week6/nutrition50_volume/dish_<id>/`.

6. **Scene Quicklooks**  
   - The first two successful masks trigger a 3D scatter preview, saved under `Week6/nutrition50_volume/quicklooks/`.  
   - These PNGs colour points by height above the inferred plate and are ideal for reports.

After iterating over all 50 dishes we also emit `Week6/nutrition50_volume/nutrition50_volume_summary.csv` containing per‑dish per‑category volumes and diagnostic stats.

---

## How The Math Works (Narrative Friendly)

1. FoodSAM gives us clean 2D silhouettes for every food item.  
2. ZoeDepth predicts a per‑pixel metric depth map.  
3. We automatically calibrate the camera by ensuring the background table projects to a flat plane.  
4. For each food mask we recover its 3D geometry relative to the plate (height map).  
5. Integrating height × pixel area yields a physically meaningful volume in millilitres.  
6. Volumes are reported both per mask and summed per food label (e.g., “cilantro mint”, “olives”).

This approach avoids guessing food thickness: the depth map tells us how high each pixel sits above the plate.

---

## Outputs You Can Cite

For any dish `dish_<id>`:

- `volumes_per_mask.csv` – one row per mask, includes metrics: `volume_m3`, `volume_ml`, `mean_height_cm`, `max_height_cm`, `support_rmse_mm`, `support_pixels`, `selected_fov_deg`.  
- `volumes_per_category.csv` – grouped by FoodSAM label (sums volumes, averages heights).  
- `volume_summary.json` – bundled record with camera intrinsics, totals, and the narrative `food_breakdown` string.  
- `quicklooks/*.png` – 3D previews (if generated) for quick visuals.

These files feed directly into downstream comparison notebooks (e.g., the LLM macro evaluation in `llms_approach/results.ipynb`).

---

## Suggested Report Outline

1. **Inputs** – show `pred_vis.png` alongside the original dish image; list ZoeDepth + mask artifacts.  
2. **Calibration** – explain the FOV sweep and plane fit with one sentence + reference the RMSE diagnostic.  
3. **Volume Extraction** – describe the support ring, height integration, and why only positive heights count.  
4. **Results** – include a table (from `volumes_per_category.csv`) and a 3D quicklook figure.  
5. **Validation** – mention comparison against Nutrition5k metadata or LLM estimates; cite `nutrition50_volume_summary.csv`.  
6. **Limitations & Next Steps** – note failure modes (insufficient background support, depth noise) and potential improvements (manual calibration target, per‑dish thickness calibration, multi‑view capture).

Keep this document synced with your write‑up and you will always have an authoritative description of the pipeline.
