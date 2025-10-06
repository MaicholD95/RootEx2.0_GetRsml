# RootEx 2.0 — Post-DL Pipeline (Graph → Paths → RSML)

> This repository documents the **post–deep learning** stage of the RootEx method: from a trained segmentation model’s output to **graph construction & refinement**, **tip→source path extraction/selection**, **optional GT evaluation**, and **RSML generation**.

- **Input**: trained model `.pth`, test images (+ per-image JSON), optional GT graph JSON.  
- **Output**: skeleton & graph visualizations, final paths, and RSML files.  
- **Scope**: everything **after** the deep network inference (the model is already trained elsewhere).

---

## ✨ Highlights

- Robust **skeleton → graph** conversion with pruning and node regularization
- Stable **tip/source anchoring** (snap to nearest graph nodes)
- Path enumeration and **optimal selection** with multi-term cost (IoU/novelty/tortuosity)
- **GT-optional**: runs metrics if GT is present; otherwise selects defaults and skips metrics
- Exports **RSML** per image

---

## 📦 Requirements

- Python 3.10+ (3.12 recommended)
- PyTorch (CUDA optional but recommended for inference)
- NumPy, OpenCV-Python, NetworkX, Matplotlib

Install the basics (adapt to your environment):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu
pip install numpy opencv-python networkx matplotlib
```

> If your project has a dedicated `requirements.txt` or `environment.yml`, prefer that.

---

## 🗂️ Project Layout (post-DL bits)

```
.
├─ main.py                         # Entry point for the post-DL pipeline
├─ dataset/                        # CustomRGBDataset (test loader)
├─ post_process_predicted_mask.py  # Predictor wrapper around the trained model (.pth)
├─ Skeleton/
│  └─ skeleton.py                  # Skeletonization utilities
├─ Graph/
│  ├─ graph_processing.py          # pruning, snapping, etc.
│  ├─ graph_utils.py               # equidistant nodes, helpers
│  └─ visualization.py             # graph overlays, debugging views
├─ Pwalking/
│  └─ path_walking_new.py          # path enumeration + valid paths
├─ gt_comparison/
│  └─ compare_with_gt.py           # GT-optional evaluation & metrics
├─ Rsml/
│  └─ create_rsml.py               # RSML writer
└─ ...
```

> Names may vary slightly in your repo; the README explains how the pieces connect.

---

## ⚙️ Configuration

Edit the **paths** and **thresholds** near the bottom of `main.py`.

### Required paths

- `model_path`: trained checkpoint `.pth`
- `dataset_path`: directory with test images + per-image JSON
- `gt_graph_folder_path` (optional): GT graph JSON (if present, metrics are computed)
- Output folders (created automatically if missing):
  - `skeletons_saving_path/`
  - `overlapped_graphs_path/`
  - `final_paths_folder/`
  - `rsml_output_folder/`

### Predictor (post-processing) parameters

```python
predictor = Predictor(
    model_path, device=device,
    resize_height=1400, resize_width=1200,
    root_threshold=0.45, tip_threshold=0.50, source_threshold=0.53,
    sigma=15,               # heatmap→keypoint smoothing
    area_threshold=320,     # remove tiny blobs
    circle_radius=20,       # NMS / peak separation
    spacing_radius=18       # enforce spacing between tips
)
```

Tune these to your data scale. Thresholds control binarization for roots/tips/sources; `sigma` & radii affect tip/source separation and stability.

---

## ▶️ How to Run

From the project root:

```bash
python main.py
```

You should see logs like:

- number of test images
- whether GT is found (→ metrics) or not (→ GT-optional mode)
- per-image RSML export confirmation

Outputs will populate the configured folders.

---

## 🔬 Pipeline (per-image)

1. **Inference & masks → PlantImage**  
   `predictor.predict_and_visualize(...)` builds a `PlantImage` with predicted masks and basic visualizations.

2. **Skeletonization & initial graph**  
   - Skeleton from root mask (`get_skeleton`), saved as `skeleton_<name>.png`
   - Pruned graph via `get_pruned_skeleton_graph(...)` (merges close nodes, removes spurs, etc.)

3. **Graph refinement**  
   - **Equidistant interpolation**: `divide_paths_with_equidistant_nodes(G, step=40)` to regularize arc length  
   - **Snap anchors**: move tips/sources to nearest nodes for stability:  
     ```python
     plant_img = move_points_to_nearest_node(plant_img, 150, 100)  # tip_radius=150, source_radius=100
     ```
     > Use **positional** args; some versions don’t accept `max_tip_dist=` style keywords.
   - **Heuristic extra tips**: mark long leaf ends not labeled as tips  
   - **Remove isolated** subgraphs not connected to any source  
   - Save intermediate visualizations and an **overlay** on the original image

4. **Path enumeration & selection**  
   - Enumerate candidate tip→source paths (cycle-free, angular constraints)  
   - Select a **non-overlapping, plausible** set by minimizing a multi-term cost combining:
     - path **IoU / novelty** (cover new nodes, avoid duplicates)
     - **tortuosity** regularization
     - optional path length / smoothness terms

5. **(Optional) GT comparison**  
   - If GT graphs are provided, evaluate a parameter grid and compute metrics; otherwise, pick a default setting and **skip** metrics.

6. **RSML export**  
   - Convert final paths to coordinates (tip→source as required) and write one **RSML** per image.

---

## 📈 Metrics (with GT)

When GT is present, the pipeline aggregates and prints:

- **Keypoints (tips/sources)**: assignment quality (e.g., distance-bounded matches, FNR/FPR)
- **Paths / RSA**: path-level distance measures (e.g., DTW of polylines with optimal matching)
- **Plant-/root-level traits** (if included in your evaluation scripts): total length, max length, tortuosity, coverage, angles, etc.

If GT is **not** present, the run still produces final paths and RSML, just without metrics.

---

## 🧪 Reproducibility

`main.py` seeds Python, NumPy, and PyTorch; it sets deterministic CuDNN and disables benchmarking to reduce run-to-run variance.

---


## 📚 Citation

If you use this code or any part of the pipeline in academic work, please cite the corresponding RootEx paper (eswa link soon, pls mail me)

---

## 🔑 License

 Apache-2.0

## 🙌 Acknowledgments

- OpenCV & NetworkX for graph/skeleton utilities  
- PyTorch for the upstream deep network used to generate the masks
