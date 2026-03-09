import os
import glob
import threading
import shutil
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from PIL import Image
import numpy as np
import colorsys

# Matplotlib for image display.
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter.filedialog as filedialog


def get_image_files(directory):
    """Return a sorted list of image files in the given directory."""
    extensions = ('*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.bmp')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    return sorted(files)


def determine_imagedata_type(pil_image):
    """Return the PIL image mode."""
    return pil_image.mode


def parse_z_value(dir_path):
    """Parse the directory's basename as a float (e.g., '-17.000' -> -17.0)."""
    try:
        return float(os.path.basename(dir_path))
    except ValueError:
        return None


def rgb_to_hex(rgb):
    """Convert an (r, g, b) tuple (each 0–1) to a hex color string."""
    return '#' + ''.join(f'{int(max(0, min(1, c))*255):02x}' for c in rgb)


def get_track_color(track_index):
    """
    Return a color for the given track index.
    For indices 0–19, use one of 20 predefined distinct colors.
    For indices 20–49, generate additional colors using a golden ratio–based approach.
    """
    if track_index < 20:
        distinct_colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#ff00ff'
        ]
        return distinct_colors[track_index]
    else:
        golden_ratio_conjugate = 0.61803398875
        hue = ((track_index - 20) * golden_ratio_conjugate) % 1
        saturation = 0.8
        value = 0.8
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        return rgb_to_hex((r, g, b))


class ImageTrackerApp:
    """
    An image tracker application with the following features:
      1. Tracks are named T1, T2, T3, etc.
      2. The left/right arrow keys navigate through the time index.
      3. The up/down arrow keys adjust the z–plane one tick at a time.
      4. A Unit Conversion panel allows converting x, y, z, and time values.
         In the Edit Mode Track panel, x, y, and z are shown to one decimal place,
         and time is shown to three decimal places.
      5. The raw track data is stored in self.raw_tracks so that conversion is applied only once.
      6. Autosave of tracks every 60s if modified (to timestamped subdir).
      7. Duplicate-time detection on new point addition.
      8. Global min/max computed in background so UI remains responsive.
      9. unitconv.npy saved alongside track .npy files.

    UPDATE (your requested change):
      - Edit Mode Track table now shows:
          Select #, ID #, X, Y, Z, Time
      - Saved .npy files now contain exactly 5 columns:
          ID #, x, y, z, t
        where ID # is assigned by temporal ordering (smallest t => 1, then 2, 3, ...).
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Image Tracker with Binary Contrast at 100%")

        # Ask for input and output directories.
        self.input_dir = filedialog.askdirectory(title="Select Path to Input Directory")
        self.output_dir = filedialog.askdirectory(title="Select Path to Output Directory")

        all_subdirs = [
            os.path.join(self.input_dir, d)
            for d in os.listdir(self.input_dir)
            if os.path.isdir(os.path.join(self.input_dir, d)) and parse_z_value(os.path.join(self.input_dir, d)) is not None
        ]
        if not all_subdirs:
            messagebox.showerror("Error", "No properly named sub-directories found.")
            root.destroy()
            return

        self.subdirs = sorted(all_subdirs, key=lambda d: parse_z_value(d), reverse=True)
        self.images_by_subdir = {sub: get_image_files(sub) for sub in self.subdirs}
        self.sub_index = 0
        self.time_index = 0

        # Store the raw (unconverted) track points as tuples:
        #   (select_id, x_raw, y_raw, z_raw, t_raw)
        self.raw_tracks = {}
        self.edit_mode_track = None   # Only one track is in edit mode.
        self.track_order = []         # Maintained in ascending order (e.g., T1, T2, ...)
        self.row_counter = 1          # Used for "Select #" unique IDs (and treeview iids)

        self.locked_region = None
        self.last_clicked_item = None

        self.zoom_xlim = None
        self.zoom_ylim = None

        self.contrast_value = 50      # default contrast slider value
        self.show_tracks_overlay = False
        self.track_name_to_id = {}    # mapping: track name -> track id
        self.last_added_point = None
        self.selected_tracks = []     # list of track IDs selected in the All Tracks listbox

        self.last_selection_mode = "overlay"  # "overlay" vs. "edit"
        self.tracks_alpha = 0.5       # default overlay opacity

        # --- Reversal event mode/state ---
        self.mode_var = tk.StringVar(value="Tracking")   # "Tracking" or "Reversal event"
        self.reversal_window = None
        self.reversal_tree = None

        # store reversal events as list of dicts holding RAW point values
        # each event: {"track": "T2", "select_id": 7, "x":..., "y":..., "z":..., "t":...}
        self.reversal_events = []

        # set of (track_id, select_id) for fast overlay marker switching to "X"
        self.reversal_point_keys = set()

        # how close (pixels) a click must be to a point to count as selecting it
        self.reversal_pick_radius = 12


        # Unit conversion ratios (default = 1).
        self.x_ratio = 1.0
        self.y_ratio = 1.0
        self.z_ratio = 1.0
        self.t_ratio = 1.0

        self.e_key_pressed = False    # For detecting e+click.
        self.sorted_tracks = []       # Sorted list of track names

        # Ensure global_min/global_max exist before they're used elsewhere:
        self.global_min = None
        self.global_max = None

        # Autosave state
        self.autosave_job = None
        self.autosave_dir = None
        self.dirty = False

        self.build_ui()
        # start background global min/max computation
        self.precompute_global_min_max_with_progress()

        # Bind arrow keys globally so that they always adjust the z–plane.
        self.root.bind_all("<Left>", self.on_left_key)
        self.root.bind_all("<Right>", self.on_right_key)
        self.root.bind_all("<Up>", self.on_up_key)
        self.root.bind_all("<Down>", self.on_down_key)
        self.root.bind("<Delete>", lambda event: self.delete_selection())
        self.track_tree.bind("<Delete>", lambda event: self.delete_selection())
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.root.bind("<KeyPress-e>", self.on_e_key_press)
        self.root.bind("<KeyRelease-e>", self.on_e_key_release)
        self.update_image()

    # --- Autosave helpers ---
    def _ensure_autosave_scheduled(self):
        if self.autosave_job is None:
            self.autosave_job = self.root.after(60000, self.autosave_tracks)

    def _converted_points_for_saving(self, raw_points):
        """
        Returns a numpy array with exactly 5 columns:
            (ID#, x, y, z, t)
        where ID# is assigned by ascending raw time (point[4]).
        """
        if not raw_points:
            return None

        sorted_points = sorted(raw_points, key=lambda r: r[4])  # time ordering

        converted = []
        for id_num, point in enumerate(sorted_points, start=1):
            conv_x = round(point[1] * self.x_ratio, 4)
            conv_y = round(point[2] * self.y_ratio, 4)
            conv_z = round(point[3] * self.z_ratio, 4)
            conv_t = round(point[4] * self.t_ratio, 4)
            converted.append((id_num, conv_x, conv_y, conv_z, conv_t))

        return np.array(converted)

    def _track_num_from_id(self, track_id: str):
        """Convert 'T2' -> 2; returns None if not parseable."""
        try:
            return int(track_id[1:])
        except:
            return None

    def _reversal_events_array_for_saving(self):
        """
        Build a numeric array for saving reversal events.

        Output columns (Nx7):
          track_num, select_id, id_num, x, y, z, t

        x,y,z,t are saved in CONVERTED units, consistent with track saving.
        """
        if not self.reversal_events:
            return None

        rows = []
        for ev in self.reversal_events:
            track_id = ev.get("track")
            select_id = int(ev.get("select_id"))
            track_num = self._track_num_from_id(track_id)
            if track_num is None:
                continue

            # temporal ID number in that track (by increasing raw time)
            id_num = self.get_temporal_id_for_select_id(track_id, select_id)
            if id_num == "":
                id_num = 0

            conv_x = round(ev["x"] * self.x_ratio, 4)
            conv_y = round(ev["y"] * self.y_ratio, 4)
            conv_z = round(ev["z"] * self.z_ratio, 4)
            conv_t = round(ev["t"] * self.t_ratio, 4)

            rows.append((track_num, select_id, int(id_num), conv_x, conv_y, conv_z, conv_t))

        if not rows:
            return None
        return np.array(rows, dtype=np.float64)

    def _load_reversal_events_npy(self, reversal_dir, x_saved, y_saved, z_saved, t_saved):
        """
        Load reversal events from:
            reversal_dir/reversal_events.npy

        Expected columns (Nx7):
          track_num, select_id, id_num, x, y, z, t
        where x,y,z,t are in CONVERTED units saved using x_saved,y_saved,z_saved,t_saved.

        This restores self.reversal_events using RAW units and rebuilds self.reversal_point_keys.
        """
        path = os.path.join(reversal_dir, "reversal_events.npy")
        if not os.path.isfile(path):
            return False

        try:
            arr = np.load(path)
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load reversal_events.npy:\n{e}")
            return False

        if arr.ndim != 2 or arr.shape[1] != 7:
            messagebox.showwarning("Warning", "reversal_events.npy has unexpected shape; expected Nx7.")
            return False

        # clear current reversal markings
        self.reversal_events = []
        self.reversal_point_keys = set()

        existing_tracks = set(self.raw_tracks.keys())

        for row in arr:
            try:
                track_num = int(row[0])
                select_id = int(row[1])
                # id_num = int(row[2])  # present but not required for restore
                conv_x = float(row[3])
                conv_y = float(row[4])
                conv_z = float(row[5])
                conv_t = float(row[6])
            except Exception:
                continue

            track_id = f"T{track_num}"
            if track_id not in existing_tracks:
                # if track names got renumbered during conflict resolution, skip
                continue

            # convert back to RAW internal units using the saved ratios
            raw_x = conv_x / x_saved
            raw_y = conv_y / y_saved
            raw_z = conv_z / z_saved
            raw_t = conv_t / t_saved

            # verify select_id exists in loaded track
            pts = self.raw_tracks.get(track_id, [])
            if not any(p[0] == select_id for p in pts):
                # cannot map reliably; skip
                continue

            self.reversal_events.append({
                "track": track_id,
                "select_id": select_id,
                "x": raw_x,
                "y": raw_y,
                "z": raw_z,
                "t": raw_t
            })
            self.reversal_point_keys.add((track_id, select_id))

        return True



    def autosave_tracks(self):
        if self.dirty:
            # delete old autosave dir
            if self.autosave_dir and os.path.isdir(self.autosave_dir):
                shutil.rmtree(self.autosave_dir)
            # new autosave directory
            timestamp = datetime.now().strftime("%d_%m_%y %H-%M")
            dir_name = f"Autosaved_Manual_Tracks {timestamp}"
            new_dir = os.path.join(self.output_dir, dir_name)
            os.makedirs(new_dir, exist_ok=True)


            # --- NEW: create TrackData and ReversalData subdirs inside autosave dir ---
            track_dir = os.path.join(new_dir, "TrackData")
            reversal_dir = os.path.join(new_dir, "ReversalData")
            os.makedirs(track_dir, exist_ok=True)
            os.makedirs(reversal_dir, exist_ok=True)

            # save tracks into TrackData (ID#, x, y, z, t)
            for track_id, raw_points in self.raw_tracks.items():
                arr = self._converted_points_for_saving(raw_points)
                if arr is None:
                    continue
                np.save(os.path.join(track_dir, f"{track_id}.npy"), arr)

            # save unit conversion into TrackData
            np.save(
                os.path.join(track_dir, "unitconv.npy"),
                np.array([self.x_ratio, self.y_ratio, self.z_ratio, self.t_ratio])
            )

            # save reversal events into ReversalData (if any)
            rev_arr = self._reversal_events_array_for_saving()
            if rev_arr is not None:
                np.save(os.path.join(reversal_dir, "reversal_events.npy"), rev_arr)

            self.autosave_dir = new_dir
            self.dirty = False

        # schedule next autosave
        self.autosave_job = self.root.after(60000, self.autosave_tracks)

    # --- Key "e" event handlers ---
    def on_e_key_press(self, event):
        self.e_key_pressed = True

    def on_e_key_release(self, event):
        self.e_key_pressed = False

    # --- Unit Conversion Updates ---
    def update_conversion_factors(self, event=None):
        try:
            self.x_ratio = float(self.x_entry.get())
        except:
            self.x_ratio = 1.0
        try:
            self.y_ratio = float(self.y_entry.get())
        except:
            self.y_ratio = 1.0
        try:
            self.z_ratio = float(self.z_entry.get())
        except:
            self.z_ratio = 1.0
        try:
            self.t_ratio = float(self.t_entry.get())
        except:
            self.t_ratio = 1.0
        self.update_track_table()
        self.root.focus_set()

    # --- Refresh the All Tracks listbox.
    def refresh_track_listbox(self, preserve_view=False):
        if preserve_view:
            current_yview = self.track_listbox.yview()
        self.sorted_tracks = sorted(self.raw_tracks.keys(), key=lambda s: int(s[1:]))
        self.track_order = self.sorted_tracks[:]
        self.track_listbox.delete(0, tk.END)
        for i, track in enumerate(self.sorted_tracks):
            self.track_listbox.insert(tk.END, track)
            if track == self.edit_mode_track and self.last_selection_mode == "edit":
                self.track_listbox.itemconfig(i, {'bg': 'lightyellow'})
            elif track in self.selected_tracks:
                self.track_listbox.itemconfig(i, {'bg': 'lightblue'})
            else:
                self.track_listbox.itemconfig(i, {'bg': 'white'})
        if preserve_view:
            self.track_listbox.yview_moveto(current_yview[0])

    # --------------------- Global Min/Max + Contrast Logic ---------------------
    def on_contrast_scroll(self, val_str):
        self.contrast_value = int(val_str)
        self.update_image()

    def apply_contrast_transform(self, arr):
        if self.global_min is None or self.global_max is None:
            return arr
        gmin = self.global_min
        gmax = self.global_max
        mid = (gmin + gmax) / 2.0
        c_norm = (self.contrast_value - 50) / 50.0
        if c_norm <= 0:
            return arr.astype(np.float32)
        elif c_norm >= 1:
            new_arr = arr.astype(np.float32)
            new_arr[new_arr < mid] = gmin
            new_arr[new_arr >= mid] = gmax
            return new_arr
        else:
            factor = 1.0 / (1.0 - c_norm + 1e-8)
            arr_f = arr.astype(np.float32)
            new_arr = mid + factor * (arr_f - mid)
            new_arr[new_arr < gmin] = gmin
            new_arr[new_arr > gmax] = gmax
            return new_arr

    def precompute_global_min_max_with_progress(self):
        total = sum(len(get_image_files(sub)) for sub in self.subdirs)
        if total == 0:
            self.global_min = 0
            self.global_max = 255
            return
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.left_frame, variable=self.progress_var, maximum=total)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        def _background():
            gmin, gmax = None, None
            count = 0
            for sub in self.subdirs:
                for img_path in get_image_files(sub):
                    try:
                        pil_img = Image.open(img_path)
                        arr = np.array(pil_img)
                        local_min, local_max = arr.min(), arr.max()
                        if gmin is None or local_min < gmin:
                            gmin = local_min
                        if gmax is None or local_max > gmax:
                            gmax = local_max
                    except Exception as e:
                        print(f"Warning: cannot open {img_path}: {e}")
                    count += 1
                    # update progress in main thread
                    self.root.after(0, self.progress_var.set, count)
            self.global_min, self.global_max = gmin, gmax
            print(f"Global range => min={self.global_min}, max={self.global_max}")
            self.root.after(0, self.progress_bar.pack_forget)

        threading.Thread(target=_background, daemon=True).start()

    # -------------------- Selection in the Treeview (Edit Mode Data) --------------------
    def on_tree_click_normal(self, event):
        self.track_listbox.selection_clear(0, tk.END)
        region = self.track_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.track_tree.identify_row(event.y)
        if not item_id:
            return
        self.track_tree.selection_set(item_id)
        self.last_clicked_item = item_id
        self.update_locked_region()
        self.update_row_tags()
        self.jump_to_time_from_selection()
        return "break"

    def on_tree_click_shift(self, event):
        self.track_listbox.selection_clear(0, tk.END)
        region = self.track_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.track_tree.identify_row(event.y)
        if not item_id:
            return
        children = list(self.track_tree.get_children())
        if not self.last_clicked_item or (self.last_clicked_item not in children):
            self.track_tree.selection_set(item_id)
            self.last_clicked_item = item_id
        else:
            idx1 = children.index(self.last_clicked_item)
            idx2 = children.index(item_id)
            start, end = sorted([idx1, idx2])
            sel_range = children[start:end+1]
            self.track_tree.selection_set(sel_range)
        self.update_locked_region()
        self.update_row_tags()
        self.jump_to_time_from_selection()
        return "break"

    def on_tree_click_ctrl(self, event):
        self.track_listbox.selection_clear(0, tk.END)
        region = self.track_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        item_id = self.track_tree.identify_row(event.y)
        if not item_id:
            return
        cur_sel = set(self.track_tree.selection())
        if item_id in cur_sel:
            cur_sel.remove(item_id)
        else:
            cur_sel.add(item_id)
            self.last_clicked_item = item_id
        self.track_tree.selection_set(list(cur_sel))
        self.update_locked_region()
        self.update_row_tags()
        self.jump_to_time_from_selection()
        return "break"

    def update_locked_region(self):
        selected = self.track_tree.selection()
        times = []
        for iid in selected:
            vals = self.track_tree.item(iid, "values")
            # now columns are: Select#, ID#, X, Y, Z, Time  => time index is 5
            if len(vals) >= 6:
                try:
                    times.append(float(vals[5]))
                except:
                    pass
        self.locked_region = (min(times), max(times)) if times else None

    def update_row_tags(self):
        # ── preserve z‐jump tags and only toggle 'selected' in blue ──
        for iid in self.track_tree.get_children():
            current_tags = set(self.track_tree.item(iid, 'tags'))
            current_tags.discard('selected')
            if iid in self.track_tree.selection():
                current_tags.add('selected')
            self.track_tree.item(iid, tags=list(current_tags))
        self.track_tree.tag_configure('selected', background='lightblue')

    def jump_to_time_from_selection(self):
        selected = self.track_tree.selection()
        if not selected:
            return
        times = []
        for iid in selected:
            vals = self.track_tree.item(iid, "values")
            # time is now at index 5
            if len(vals) >= 6:
                try:
                    times.append(float(vals[5]))
                except:
                    pass
        if times:
            self.time_index = int(min(times))
            self.t_scroll.set(self.time_index)
            self.update_image()

    # -------------------- Scrollbars / Buttons for z & time ---------------------
    def z_prev(self, event=None):
        if self.sub_index > 0:
            self.sub_index -= 1
            self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))
            self.update_image()

    def z_next(self, event=None):
        if self.sub_index < len(self.subdirs) - 1:
            self.sub_index += 1
            self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))
            self.update_image()

    def t_prev(self):
        if self.time_index > 0:
            new_val = self.time_index - 1
            self.t_scroll.set(new_val)
            self.on_t_scroll(new_val)

    def t_next(self):
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        if self.time_index < max_t:
            new_val = self.time_index + 1
            self.t_scroll.set(new_val)
            self.on_t_scroll(new_val)

    def find_closest_subdir(self, desired, direction="up"):
        """
        Find the index and z value of the subdirectory with a z value closest to the desired.

        The search considers the 'direction':
            - For 'up': selects subdirectories with a z value greater than or equal to the desired.
            - For 'down': selects subdirectories with a z value less than or equal to the desired.
        If none match the direction criteria, the subdirectory with the smallest absolute difference is returned.

        Returns:
            (index, z_value) tuple.
        """
        best_index = None
        best_diff = float('inf')

        for i, subdir in enumerate(self.subdirs):
            z_val = parse_z_value(subdir)
            if z_val is None:
                continue
            if direction == "up" and z_val >= desired:
                diff = abs(z_val - desired)
                if diff < best_diff:
                    best_index = i
                    best_diff = diff
            elif direction == "down" and z_val <= desired:
                diff = abs(z_val - desired)
                if diff < best_diff:
                    best_index = i
                    best_diff = diff

        if best_index is None:
            for i, subdir in enumerate(self.subdirs):
                z_val = parse_z_value(subdir)
                if z_val is None:
                    continue
                diff = abs(z_val - desired)
                if diff < best_diff:
                    best_index = i
                    best_diff = diff

        actual_z = parse_z_value(self.subdirs[best_index])
        return best_index, actual_z

    def on_z_scroll(self, value, direction="up"):
        try:
            desired = float(value)
        except ValueError:
            return
        closest_index, actual_z = self.find_closest_subdir(desired, direction)
        self.sub_index = closest_index
        self.z_scroll.set(actual_z)
        self.z_value_label.config(text=f"Z: {actual_z}")
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        if self.time_index > max_t:
            self.time_index = max_t
        self.t_scroll.config(to=max_t)
        self.t_scroll.set(self.time_index)
        self.update_image()

    def on_t_scroll(self, value):
        self.time_index = int(value)
        self.update_image()

    # --- Left/Right Arrow Key Handlers for Time Navigation ---
    def on_left_key(self, event):
        if self.time_index > 0:
            self.time_index -= 1
            self.t_scroll.set(self.time_index)
            self.update_image()

    def on_right_key(self, event):
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        if self.time_index < max_t:
            self.time_index += 1
            self.t_scroll.set(self.time_index)
            self.update_image()

    # --- Up/Down Arrow Key Handlers for Z-plane Navigation ---
    def on_up_key(self, event):
        current_z = float(self.z_scroll.get())
        max_z = float(self.z_scroll.cget("to"))
        if current_z < max_z:
            new_z = current_z + 1
            self.z_scroll.set(new_z)
            self.on_z_scroll(new_z, direction="up")
        return "break"

    def on_down_key(self, event):
        current_z = float(self.z_scroll.get())
        min_z = float(self.z_scroll.cget("from"))
        if current_z > min_z:
            new_z = current_z - 1
            self.z_scroll.set(new_z)
            self.on_z_scroll(new_z, direction="down")
        return "break"

    # -------------------- Displaying the Image (with Contrast and Zoom) ---------------------
    def update_image(self):
        c_subdir = self.subdirs[self.sub_index]
        imgs = self.images_by_subdir[c_subdir]
        if not imgs:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No images in this directory", ha="center", va="center", color="white")
            self.canvas.draw()
            self.current_image_array = None
            return
        if self.time_index >= len(imgs):
            self.time_index = len(imgs) - 1
        img_path = imgs[self.time_index]
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open {img_path}\n{e}")
            return
        arr = np.array(pil_img, dtype=np.float32)
        self.current_image_array = arr
        self.ax.clear()
        arr_c = self.apply_contrast_transform(arr)
        vmin = self.global_min if self.global_min is not None else arr_c.min()
        vmax = self.global_max if self.global_max is not None else arr_c.max()
        if len(arr_c.shape) == 2:
            self.ax.imshow(arr_c, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            mode = determine_imagedata_type(pil_img)
            if mode in ["RGB", "RGBA"]:
                self.ax.imshow(arr_c.astype(np.uint8))
            else:
                self.ax.imshow(arr_c, cmap="gray", vmin=vmin, vmax=vmax)
        if self.zoom_xlim is not None and self.zoom_ylim is not None:
            self.ax.set_xlim(self.zoom_xlim)
            self.ax.set_ylim(self.zoom_ylim)
        else:
            self.zoom_xlim = self.ax.get_xlim()
            self.zoom_ylim = self.ax.get_ylim()
        if self.show_tracks_overlay:
            self.draw_all_tracks()
        self.ax.axis("off")
        self.canvas.draw()

    def on_canvas_motion(self, event):
        if self.current_image_array is None or event.xdata is None or event.ydata is None:
            self.pixel_info_label.config(text="Pixel: N/A")
            return
        x = int(event.xdata)
        y = int(event.ydata)
        h, w = self.current_image_array.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            self.pixel_info_label.config(text="Pixel: N/A")
            return
        val = self.current_image_array[y, x]
        self.pixel_info_label.config(text=f"Pixel:({x},{y}) Value:{val}")

    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        click_x = int(event.xdata)
        click_y = int(event.ydata)

        # --- Reversal event mode: click marks reversal events (does NOT add track points) ---
        if self.mode_var.get() == "Reversal event":
            self.mark_reversal_event_from_click(click_x, click_y)
            return

        # --- Tracking mode (existing behavior) ---
        if self.edit_mode_track is None:
            if not self.raw_tracks:
                self.start_new_track()
            else:
                return

        x = click_x
        y = click_y
        z = parse_z_value(self.subdirs[self.sub_index])
        t = self.time_index

        track_pts = self.raw_tracks[self.edit_mode_track]
        existing = [pt for pt in track_pts if pt[4] == t]
        if existing:
            replace = messagebox.askyesno(
                "Duplicate Time",
                f"Time point {t} already exists. Replace old data point?"
            )
            if replace:
                track_pts[:] = [pt for pt in track_pts if pt[4] != t]
            else:
                return

        raw_point = (self.row_counter, x, y, z, t)  # (Select#, x, y, z, t)
        track_pts.append(raw_point)
        self.last_added_point = (x, y, z, t)
        self.row_counter += 1
        self.update_track_table()
        self.jump_to_next_time()

        self.dirty = True
        self._ensure_autosave_scheduled()

    def jump_to_next_time(self):
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        new_t = self.time_index + 1
        if new_t > max_t:
            new_t = max_t
        self.time_index = new_t
        self.t_scroll.set(new_t)
        self.update_image()

    def on_scroll(self, event):
        if event.key is None or "control" not in event.key.lower():
            return
        base_scale = 1.2
        if event.button == 'up':
            scale_factor = 1 / base_scale
        elif event.button == 'down':
            scale_factor = base_scale
        else:
            scale_factor = 1
        cur_xlim = self.zoom_xlim if self.zoom_xlim is not None else self.ax.get_xlim()
        cur_ylim = self.zoom_ylim if self.zoom_ylim is not None else self.ax.get_ylim()
        if event.xdata is None or event.ydata is None:
            return
        xdata = event.xdata
        ydata = event.ydata
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        new_xmin = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_xmax = new_xmin + new_width
        new_ymin = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_ymax = new_ymin + new_height
        self.zoom_xlim = (new_xmin, new_xmax)
        self.zoom_ylim = (new_ymin, new_ymax)
        self.ax.set_xlim(self.zoom_xlim)
        self.ax.set_ylim(self.zoom_ylim)
        self.canvas.draw()

    def on_ctrl_mousewheel(self, event):
        base_scale = 1.2
        if event.delta > 0:
            scale_factor = 1 / base_scale
        else:
            scale_factor = base_scale
        widget = self.canvas.get_tk_widget()
        height = widget.winfo_height()
        x = event.x
        y = height - event.y
        try:
            xdata, ydata = self.ax.transData.inverted().transform((x, y))
        except Exception:
            return "break"
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        new_xmin = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_xmax = new_xmin + new_width
        new_ymin = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_ymax = new_ymin + new_height
        self.ax.set_xlim(new_xmin, new_xmax)
        self.ax.set_ylim(new_ymin, new_ymax)
        self.zoom_xlim = (new_xmin, new_xmax)
        self.zoom_ylim = (new_ymin, new_ymax)
        self.canvas.draw()
        return "break"

    def reset_zoom(self):
        self.zoom_xlim = None
        self.zoom_ylim = None
        self.update_image()

    def update_track_table(self):
        if self.edit_mode_track is None:
            return

        raw_points = self.raw_tracks[self.edit_mode_track]
        sorted_points = sorted(raw_points, key=lambda r: r[4])  # temporal ordering (raw time)

        for iid in self.track_tree.get_children():
            self.track_tree.delete(iid)

        # build list of all converted Z’s (for z-jump highlighting)
        conv_z_list = [round(pt[3] * self.z_ratio, 1) for pt in sorted_points]

        recent_item_id = None

        # Insert each row with:
        #   Select#, ID#, X, Y, Z, Time
        for idx, point in enumerate(sorted_points):
            select_id = point[0]
            id_num = idx + 1  # temporal ID based on sorted time

            conv_x = round(point[1] * self.x_ratio, 1)
            conv_y = round(point[2] * self.y_ratio, 1)
            conv_z = conv_z_list[idx]
            conv_t = round(point[4] * self.t_ratio, 3)

            # compute z-jump tags
            diff_prev = abs(conv_z - conv_z_list[idx - 1]) if idx > 0 else 0
            diff_next = abs(conv_z - conv_z_list[idx + 1]) if idx < len(conv_z_list) - 1 else 0
            max_diff = max(diff_prev, diff_next)
            tags = []
            if max_diff > 5:
                tags.append("z_jump_high")
            elif max_diff > 3:
                tags.append("z_jump_med")

            # selected-region tag (locked_region is in display-time units)
            if self.locked_region:
                t_min, t_max = self.locked_region
                if t_min <= conv_t <= t_max:
                    tags.append("selected")

            # recently-added tag (compare raw)
            if self.last_added_point and (point[1], point[2], point[3], point[4]) == self.last_added_point:
                tags.append("recent")

            disp_point = (select_id, id_num, conv_x, conv_y, conv_z, f"{conv_t:.3f}")
            item_id = self.track_tree.insert(
                "", tk.END,
                iid=str(select_id),   # iid remains Select# so deletion keeps working
                values=disp_point,
                tags=tags
            )

            if "recent" in tags:
                recent_item_id = item_id

        # configure tags
        self.track_tree.tag_configure("selected", background="lightgray")
        self.track_tree.tag_configure("recent", font=("TkDefaultFont", 9, "bold"), background="lightyellow")
        self.track_tree.tag_configure("z_jump_med", background="orange")
        self.track_tree.tag_configure("z_jump_high", background="red")

        if recent_item_id:
            children = self.track_tree.get_children()
            total_rows = len(children)
            try:
                first_bbox = self.track_tree.bbox(children[0])
                visible_count = (self.track_tree.winfo_height() // first_bbox[3]) if first_bbox and first_bbox[3] > 0 else 10
            except Exception:
                visible_count = 10
            recent_index = children.index(recent_item_id)
            if total_rows > visible_count:
                fraction = max(0, min((recent_index - (visible_count / 2)) / (total_rows - visible_count), 1))
                self.track_tree.yview_moveto(fraction)
            else:
                self.track_tree.see(recent_item_id)

    def start_new_track(self):
        n = 1
        while f"T{n}" in self.raw_tracks:
            n += 1
        track_id = f"T{n}"
        self.raw_tracks[track_id] = []
        self.edit_mode_track = track_id
        self.track_order.append(track_id)
        self.track_info_label.config(text=f"Edit Mode Track: {track_id}")
        for iid in self.track_tree.get_children():
            self.track_tree.delete(iid)
        self.locked_region = None
        self.row_counter = 1
        self.track_name_to_id[track_id] = track_id
        self.refresh_track_listbox()

    def save_tracks(self):
        if not self.raw_tracks:
            return

        # manual save: cancel autosave timer and delete old autosave directory
        if self.autosave_job:
            self.root.after_cancel(self.autosave_job)
            self.autosave_job = None
        if self.autosave_dir and os.path.isdir(self.autosave_dir):
            shutil.rmtree(self.autosave_dir)
            self.autosave_dir = None

        # --- NEW: create TrackData and ReversalData subdirectories ---
        track_dir = os.path.join(self.output_dir, "TrackData")
        reversal_dir = os.path.join(self.output_dir, "ReversalData")
        os.makedirs(track_dir, exist_ok=True)
        os.makedirs(reversal_dir, exist_ok=True)

        # --- save track npy files into TrackData ---
        for track_id, raw_points in self.raw_tracks.items():
            arr = self._converted_points_for_saving(raw_points)
            if arr is None:
                continue
            out_path = os.path.join(track_dir, f"{track_id}.npy")
            try:
                np.save(out_path, arr)
                print(f"Saved track {track_id} => {out_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save {track_id}:\n{e}")

        # --- save unit conversion into TrackData ---
        unit_path = os.path.join(track_dir, "unitconv.npy")
        try:
            np.save(unit_path, np.array([self.x_ratio, self.y_ratio, self.z_ratio, self.t_ratio]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save unit conversion:\n{e}")

        # --- save reversal events into ReversalData (if any) ---
        rev_arr = self._reversal_events_array_for_saving()
        if rev_arr is not None:
            rev_path = os.path.join(reversal_dir, "reversal_events.npy")
            try:
                np.save(rev_path, rev_arr)
                print(f"Saved reversal events => {rev_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save reversal events:\n{e}")

        self.dirty = False

    # -------------------- Listbox Selection Handlers --------------------
    def on_listbox_click(self, event):
        index = self.track_listbox.nearest(event.y)
        if index < 0 or index >= len(self.sorted_tracks):
            return
        track_id = self.sorted_tracks[index]
        if self.e_key_pressed:
            self.last_selection_mode = "edit"
            self.edit_mode_track = track_id
            self.selected_tracks = [track_id]
            self.track_info_label.config(text=f"Edit Mode Track: {track_id}")
            self.refresh_track_listbox(preserve_view=True)
            self.update_track_table()
            return "break"

    def on_select_track(self, event):
        if not self.e_key_pressed:
            self.last_selection_mode = "overlay"
        selection = self.track_listbox.curselection()
        if not selection:
            self.selected_tracks = []
            return

        chosen = [self.track_listbox.get(i) for i in selection]

        # If in reversal mode, force single-track selection (take the last selected)
        if self.mode_var.get() == "Reversal event" and len(chosen) > 1:
            chosen = [chosen[-1]]
            self.track_listbox.selection_clear(0, tk.END)
            # reselect the last one
            last_track = chosen[0]
            try:
                idx = self.sorted_tracks.index(last_track)
                self.track_listbox.selection_set(idx)
            except:
                pass

        self.selected_tracks = chosen
        self.update_image()

    def on_listbox_ctrl_click(self, event):
        index = self.track_listbox.nearest(event.y)
        if index < 0 or index >= len(self.sorted_tracks):
            return
        track_id = self.sorted_tracks[index]
        if track_id in self.selected_tracks:
            self.track_listbox.selection_clear(index)
        else:
            self.track_listbox.selection_set(index)
        self.selected_tracks = [self.track_listbox.get(i) for i in self.track_listbox.curselection()]
        self.refresh_track_listbox(preserve_view=True)
        self.update_image()
        return "break"

    def toggle_show_tracks(self):
        self.show_tracks_overlay = not self.show_tracks_overlay
        self.show_tracks_button.config(text="Hide Tracks" if self.show_tracks_overlay else "Show Tracks")
        self.update_image()

    def draw_all_tracks(self):
        for i, track_id in enumerate(self.track_order):
            pts = self.raw_tracks.get(track_id, [])
            if not pts:
                continue

            pts_sorted = sorted(pts, key=lambda r: r[4])
            color = get_track_color(i)

            if track_id in self.selected_tracks:
                linewidth = 3
                markeredgewidth = 3
            else:
                linewidth = 1
                markeredgewidth = 1

            xvals = [p[1] for p in pts_sorted]
            yvals = [p[2] for p in pts_sorted]

            # 1) draw line
            self.ax.plot(
                xvals, yvals,
                linewidth=linewidth,
                color=color,
                linestyle='-',
                alpha=self.tracks_alpha
            )

            # 2) split points into normal vs reversal
            normal_x, normal_y = [], []
            rev_x, rev_y = [], []

            for p in pts_sorted:
                select_id = p[0]
                key = (track_id, select_id)
                if key in self.reversal_point_keys:
                    rev_x.append(p[1])
                    rev_y.append(p[2])
                else:
                    normal_x.append(p[1])
                    normal_y.append(p[2])

            # normal circles
            if normal_x:
                self.ax.scatter(
                    normal_x, normal_y,
                    marker='o',
                    facecolors='none',
                    edgecolors=color,
                    linewidths=markeredgewidth,
                    alpha=self.tracks_alpha
                )

            # reversal X markers
            if rev_x:
                self.ax.scatter(
                    rev_x, rev_y,
                    marker='x',
                    c=color,
                    linewidths=markeredgewidth + 1,
                    alpha=self.tracks_alpha
                )

    def load_previous_tracks(self):
        load_dir = filedialog.askdirectory(title="Select directory containing NPY track files")
        if not load_dir:
            return
        # --- NEW: support TrackData/ and ReversalData/ structure ---
        track_dir = os.path.join(load_dir, "TrackData")
        parent_dir = os.path.dirname(load_dir)
        reversal_dir = os.path.join(parent_dir, "ReversalData")

        # If TrackData exists, load tracks from there; otherwise fall back to old behavior
        if os.path.isdir(track_dir):
            npy_files = glob.glob(os.path.join(track_dir, "*.npy"))
        else:
            npy_files = glob.glob(os.path.join(load_dir, "*.npy"))

        if not npy_files:
            return

        # ─── load unit conversion ratios (either name) and remove them from tracks list ───
        x_saved = y_saved = z_saved = t_saved = 1.0
        for fname in ("unitconv.npy", "unitconvert.npy"):
            # Prefer TrackData/unitconv.npy if TrackData exists
            if os.path.isdir(track_dir):
                p = os.path.join(track_dir, fname)
            else:
                p = os.path.join(load_dir, fname)

            if os.path.isfile(p):
                try:
                    x_saved, y_saved, z_saved, t_saved = np.load(p)
                except Exception as e:
                    messagebox.showwarning("Warning", f"Could not load unit conversion file {fname}:\n{e}")
                break

        # remove ratio files from list
        npy_files = [
            f for f in npy_files
            if os.path.basename(f).lower() not in ("unitconv.npy", "unitconvert.npy")
        ]
        if not npy_files:
            return

        loaded_tracks = {}
        for file in npy_files:
            track_name = os.path.splitext(os.path.basename(file))[0]
            try:
                data = np.load(file)
                if data.ndim != 2 or data.shape[1] != 5:
                    messagebox.showerror("Error", f"{file} does not look like a 5-column track file.")
                    continue

                # file format now: (ID#, x, y, z, t)
                # we IGNORE file ID# and rebuild Select# internally
                temp_points = []
                for _id, x, y, z, t in data:
                    temp_points.append((
                        0,  # placeholder Select#, assigned below
                        x / x_saved,
                        y / y_saved,
                        z / z_saved,
                        t / t_saved
                    ))

                # assign Select# as unique integers after sorting by time
                temp_points_sorted = sorted(temp_points, key=lambda r: r[4])
                rebuilt = []
                for select_id, pt in enumerate(temp_points_sorted, start=1):
                    rebuilt.append((select_id, pt[1], pt[2], pt[3], pt[4]))

                loaded_tracks[track_name] = rebuilt

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file}:\n{e}")

        def extract_number(name):
            try:
                return int(name[1:])
            except:
                return None

        conflicts = []
        for track_name in loaded_tracks.keys():
            if track_name in self.track_name_to_id:
                conflicts.append(track_name)

        max_num = 0
        for name in list(self.track_name_to_id.keys()) + list(loaded_tracks.keys()):
            num = extract_number(name)
            if num and num > max_num:
                max_num = num

        for conflict in conflicts:
            track_id = self.track_name_to_id[conflict]
            max_num += 1
            new_name = f"T{max_num}"
            del self.track_name_to_id[conflict]
            self.track_name_to_id[new_name] = track_id

        for track_name, track_points in loaded_tracks.items():
            if track_name in self.track_name_to_id:
                track_id = self.track_name_to_id[track_name]
                self.raw_tracks[track_id] = track_points
            else:
                track_id = f"T{len(self.raw_tracks) + 1}"
                self.raw_tracks[track_id] = track_points
                self.track_order.append(track_id)
                self.track_name_to_id[track_name] = track_id

        self.refresh_track_listbox()

        # --- NEW: attempt to load reversal events from ReversalData/reversal_events.npy ---
        loaded_reversals = False
        if os.path.isdir(reversal_dir):
            loaded_reversals = self._load_reversal_events_npy(reversal_dir, x_saved, y_saved, z_saved, t_saved)

        if loaded_reversals:
            # show reversal window and populate table so user sees it loaded
            self.mode_var.set("Reversal event")
            self.show_reversal_window()
            self.refresh_reversal_table()
            self.update_image()

        self.dirty = True
        self._ensure_autosave_scheduled()

    def delete_selection(self):
        selected_rows = self.track_tree.selection()
        if selected_rows:
            confirm = messagebox.askokcancel(
                "Confirm Deletion",
                f"Are you sure you want to delete the selected {len(selected_rows)} row(s)?"
            )
            if not confirm:
                return
            row_ids = [int(iid) for iid in selected_rows]  # iids are Select#
            self.raw_tracks[self.edit_mode_track] = [
                pt for pt in self.raw_tracks[self.edit_mode_track]
                if pt[0] not in row_ids
            ]
            self.update_track_table()
            self.update_image()
            self.dirty = True
            self._ensure_autosave_scheduled()
            return

        selected_indices = self.track_listbox.curselection()
        if selected_indices:
            if len(selected_indices) > 1:
                confirm = messagebox.askokcancel(
                    "Confirm Deletion",
                    f"Are you sure you want to delete the selected {len(selected_indices)} tracks?"
                )
            else:
                track_name = self.track_listbox.get(selected_indices[0])
                confirm = messagebox.askokcancel(
                    "Confirm Deletion",
                    f"Are you sure you want to delete track {track_name}?"
                )
            if not confirm:
                return
            for i in reversed(selected_indices):
                track_name = self.track_listbox.get(i)
                track_id = self.track_name_to_id.get(track_name)
                if track_id in self.raw_tracks:
                    del self.raw_tracks[track_id]
                if track_id in self.track_order:
                    self.track_order.remove(track_id)
                if track_name in self.track_name_to_id:
                    del self.track_name_to_id[track_name]
                self.track_listbox.delete(i)
                if self.edit_mode_track == track_id:
                    self.edit_mode_track = None
                    self.track_info_label.config(text="Edit Mode Track:")
                    for item in self.track_tree.get_children():
                        self.track_tree.delete(item)
            self.refresh_track_listbox()
            self.update_image()
            self.dirty = True
            self._ensure_autosave_scheduled()
            return

        return

    def build_ui(self):
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_motion)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

        self.pixel_info_frame = tk.Frame(self.left_frame, height=25)
        self.pixel_info_frame.pack_propagate(False)
        self.pixel_info_frame.pack(side=tk.TOP, fill=tk.X)
        self.pixel_info_label = tk.Label(self.pixel_info_frame, text="Pixel: N/A", anchor="w")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5, pady=2)

        self.scrollbars_frame = tk.Frame(self.left_frame)
        self.scrollbars_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.z_scroll_container = tk.Frame(self.scrollbars_frame)
        self.z_scroll_container.pack(fill=tk.X, pady=2)
        self.z_values = [parse_z_value(d) for d in self.subdirs]
        min_z = min(self.z_values)
        max_z = max(self.z_values)
        self.z_scroll = tk.Scale(
            self.z_scroll_container,
            from_=min_z,
            to=max_z,
            orient=tk.HORIZONTAL,
            label="Z-plane",
            resolution=1,
            command=self.on_z_scroll
        )
        self.z_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))
        self.z_value_label = tk.Label(self.z_scroll_container, text=f"Z: {parse_z_value(self.subdirs[self.sub_index])}")
        self.z_value_label.pack(side=tk.LEFT, padx=5)
        self.z_prev_button = tk.Button(self.z_scroll_container, text="◄", command=self.z_next)
        self.z_prev_button.pack(side=tk.LEFT)
        self.z_next_button = tk.Button(self.z_scroll_container, text="►", command=self.z_prev)
        self.z_next_button.pack(side=tk.LEFT)

        current_subdir = self.subdirs[self.sub_index]
        max_time = max(len(self.images_by_subdir[current_subdir]) - 1, 0)
        self.t_scroll_container = tk.Frame(self.scrollbars_frame)
        self.t_scroll_container.pack(fill=tk.X, pady=2)
        self.t_scroll = tk.Scale(
            self.t_scroll_container,
            from_=0,
            to=max_time,
            orient=tk.HORIZONTAL,
            label="Time Series (Image Index)",
            command=self.on_t_scroll
        )
        self.t_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.t_prev_button = tk.Button(self.t_scroll_container, text="◄", command=self.t_prev)
        self.t_prev_button.pack(side=tk.LEFT)
        self.t_next_button = tk.Button(self.t_scroll_container, text="►", command=self.t_next)
        self.t_next_button.pack(side=tk.LEFT)

        self.contrast_slider = tk.Scale(
            self.left_frame,
            from_=50,
            to=100,
            orient=tk.HORIZONTAL,
            label="Contrast",
            command=self.on_contrast_scroll
        )
        self.contrast_slider.set(50)
        self.contrast_slider.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.track_info_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, width=420)
        self.track_info_frame.pack_propagate(False)
        self.track_info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        self.current_track_frame = tk.Frame(self.track_info_frame)
        self.current_track_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.unit_conv_frame = tk.Frame(self.track_info_frame, bd=1, relief=tk.GROOVE)
        self.unit_conv_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        conv_label = tk.Label(self.unit_conv_frame, text="Unit Conversion", font=("Helvetica", 14, "bold"))
        conv_label.grid(row=0, column=0, columnspan=4, pady=(2, 5))

        tk.Label(self.unit_conv_frame, text="X Ratio:").grid(row=1, column=0, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Y Ratio:").grid(row=1, column=1, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Z Ratio:").grid(row=1, column=2, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Time Ratio:").grid(row=1, column=3, sticky="e", padx=2)

        self.x_entry = tk.Entry(self.unit_conv_frame, width=5)
        self.x_entry.insert(0, "1.0")
        self.x_entry.grid(row=2, column=0, padx=2, pady=2)

        self.y_entry = tk.Entry(self.unit_conv_frame, width=5)
        self.y_entry.insert(0, "1.0")
        self.y_entry.grid(row=2, column=1, padx=2, pady=2)

        self.z_entry = tk.Entry(self.unit_conv_frame, width=5)
        self.z_entry.insert(0, "1.0")
        self.z_entry.grid(row=2, column=2, padx=2, pady=2)

        self.t_entry = tk.Entry(self.unit_conv_frame, width=5)
        self.t_entry.insert(0, "1.0")
        self.t_entry.grid(row=2, column=3, padx=2, pady=2)

        self.x_entry.bind("<Return>", self.update_conversion_factors)
        self.y_entry.bind("<Return>", self.update_conversion_factors)
        self.z_entry.bind("<Return>", self.update_conversion_factors)
        self.t_entry.bind("<Return>", self.update_conversion_factors)

        self.x_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.y_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.z_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.t_entry.bind("<FocusOut>", self.update_conversion_factors)

        self.made_tracks_frame = tk.Frame(self.track_info_frame)
        self.made_tracks_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.right_button_frame = tk.Frame(self.track_info_frame)
        self.right_button_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, pady=5)

        self.track_info_label = tk.Label(
            self.current_track_frame,
            text="Edit Mode Track:",
            anchor="w",
            font=("Helvetica", 16, "bold")
        )
        self.track_info_label.pack(fill=tk.X, padx=5, pady=5)

        # UPDATED COLUMNS:
        #   Select#, ID#, X, Y, Z, Time
        self.columns = ("select", "id", "x", "y", "z", "t")
        self.track_tree = ttk.Treeview(
            self.current_track_frame,
            columns=self.columns,
            show="headings",
            selectmode="none",
            takefocus=0
        )
        self.track_tree.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

        self.track_tree.heading("select", text="Select #")
        self.track_tree.heading("id", text="ID #")
        self.track_tree.heading("x", text="X")
        self.track_tree.heading("y", text="Y")
        self.track_tree.heading("z", text="Z")
        self.track_tree.heading("t", text="Time")

        self.track_tree.column("select", width=70, anchor=tk.CENTER)
        self.track_tree.column("id", width=60, anchor=tk.CENTER)
        self.track_tree.column("x", width=70, anchor=tk.CENTER)
        self.track_tree.column("y", width=70, anchor=tk.CENTER)
        self.track_tree.column("z", width=70, anchor=tk.CENTER)
        self.track_tree.column("t", width=80, anchor=tk.CENTER)

        style = ttk.Style()
        style.configure("selected.Table", background="lightgray")

        self.track_tree.bind("<Button-1>", self.on_tree_click_normal)
        self.track_tree.bind("<Shift-Button-1>", self.on_tree_click_shift, add="+")
        self.track_tree.bind("<Control-Button-1>", self.on_tree_click_ctrl, add="+")

        track_list_label = tk.Label(self.made_tracks_frame, text="All Tracks:", anchor="w", font=("Helvetica", 16, "bold"))
        track_list_label.pack(fill=tk.X, padx=5, pady=5)

        self.track_listbox = tk.Listbox(self.made_tracks_frame, selectmode=tk.EXTENDED, takefocus=0)
        self.track_listbox.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)
        self.track_listbox.bind("<Button-1>", self.on_listbox_click)
        self.track_listbox.bind("<<ListboxSelect>>", self.on_select_track)
        self.track_listbox.bind("<Control-Button-1>", self.on_listbox_ctrl_click)

        # prevent Up/Down from changing listbox selection
        self.track_listbox.bind("<Up>", lambda e: self.on_up_key(e))
        self.track_listbox.bind("<Down>", lambda e: self.on_down_key(e))


        # --- Mode dropdown (Tracking vs Reversal event) ---
        mode_frame = tk.Frame(self.right_button_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))
        mode_frame.columnconfigure(1, weight=1)

        tk.Label(mode_frame, text="Mode:", font=("Helvetica", 14, "bold")).grid(row=0, column=0, sticky="w")

        mode_menu = tk.OptionMenu(
            mode_frame,
            self.mode_var,
            "Tracking",
            "Reversal event",
            command=self.on_mode_change
        )
        mode_menu.config(font=("Helvetica", 12))
        mode_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))



        button_font = ("Helvetica", 16, "bold")
        self.new_track_button = tk.Button(self.right_button_frame, text="Start New Track", font=button_font, command=self.start_new_track)
        self.save_button = tk.Button(self.right_button_frame, text="Save", font=button_font, command=self.save_tracks)
        self.show_tracks_button = tk.Button(self.right_button_frame, text="Show Tracks", font=button_font, command=self.toggle_show_tracks)
        self.home_button = tk.Button(self.right_button_frame, text="🏠 Home", font=button_font, command=self.reset_zoom)
        self.load_tracks_button = tk.Button(self.right_button_frame, text="Load Previous Tracks", font=button_font, command=self.load_previous_tracks)
        self.delete_button = tk.Button(self.right_button_frame, text="Delete", font=button_font, command=self.delete_selection)
        self.translucence_slider = tk.Scale(
            self.right_button_frame,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            label="Track Opacity (%)",
            command=self.on_translucence_change,
            font=("Helvetica", 14)
        )
        self.translucence_slider.set(50)

        self.right_button_frame.columnconfigure(0, weight=1)
        self.right_button_frame.columnconfigure(1, weight=1)
        self.new_track_button.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.save_button.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.show_tracks_button.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.home_button.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.load_tracks_button.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.delete_button.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
        self.translucence_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        self.canvas.get_tk_widget().bind("<Control-MouseWheel>", self.on_ctrl_mousewheel)


    # -------------------- Reversal Mode UI + Data --------------------
    def on_mode_change(self, selected_value=None):
        """
        Called when the dropdown changes.
        """
        mode = self.mode_var.get()
        if mode == "Reversal event":
            self.show_reversal_window()
        else:
            # tracking mode
            # keep reversal data, just stop interpreting clicks as reversal events
            pass

        # refresh overlay if visible so "X" markers show if needed
        self.update_image()

    def show_reversal_window(self):
        """
        Create (if needed) and show the reversal event window.
        """
        if self.reversal_window is None or not tk.Toplevel.winfo_exists(self.reversal_window):
            self.reversal_window = tk.Toplevel(self.root)
            self.reversal_window.title("Reversal Events")
            self.reversal_window.geometry("500x300")

            # closing should just hide (withdraw) so user can bring it back
            self.reversal_window.protocol("WM_DELETE_WINDOW", self.hide_reversal_window)

            cols = ("track", "id", "x", "y", "z", "t")
            self.reversal_tree = ttk.Treeview(self.reversal_window, columns=cols, show="headings")
            self.reversal_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            self.reversal_tree.heading("track", text="Track")
            self.reversal_tree.heading("id", text="ID #")
            self.reversal_tree.heading("x", text="X")
            self.reversal_tree.heading("y", text="Y")
            self.reversal_tree.heading("z", text="Z")
            self.reversal_tree.heading("t", text="Time")

            self.reversal_tree.column("track", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("id", width=55, anchor=tk.CENTER)
            self.reversal_tree.column("x", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("y", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("z", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("t", width=90, anchor=tk.CENTER)

            btn_frame = tk.Frame(self.reversal_window)
            btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

            tk.Button(btn_frame, text="Delete Selected Event", command=self.delete_selected_reversal_event).pack(side=tk.LEFT)
            tk.Button(btn_frame, text="Hide", command=self.hide_reversal_window).pack(side=tk.RIGHT)

        # show it
        self.reversal_window.deiconify()
        self.reversal_window.lift()
        self.refresh_reversal_table()

    def hide_reversal_window(self):
        if self.reversal_window and tk.Toplevel.winfo_exists(self.reversal_window):
            self.reversal_window.withdraw()

    def refresh_reversal_table(self):
        """
        Rebuild the reversal events table from self.reversal_events.
        Values displayed are converted using x/y/z/t ratios.
        """
        if not self.reversal_tree:
            return

        # clear
        for iid in self.reversal_tree.get_children():
            self.reversal_tree.delete(iid)

        # display sorted by (track, t)
        def sort_key(ev):
            return (ev["track"], ev["t"])

        for idx, ev in enumerate(sorted(self.reversal_events, key=sort_key), start=1):
            track = ev["track"]
            select_id = ev["select_id"]

            # compute the temporal ID# based on track ordering by time
            id_num = self.get_temporal_id_for_select_id(track, select_id)

            conv_x = round(ev["x"] * self.x_ratio, 1)
            conv_y = round(ev["y"] * self.y_ratio, 1)
            conv_z = round(ev["z"] * self.z_ratio, 1)
            conv_t = round(ev["t"] * self.t_ratio, 3)

            self.reversal_tree.insert(
                "",
                tk.END,
                iid=f"{track}:{select_id}",
                values=(track, id_num, conv_x, conv_y, conv_z, f"{conv_t:.3f}")
            )

    def get_temporal_id_for_select_id(self, track_id, select_id):
        """
        Track table ID# is assigned by sorting raw points by time.
        Return the temporal ID# for the given Select# in that track.
        """
        pts = self.raw_tracks.get(track_id, [])
        if not pts:
            return ""

        pts_sorted = sorted(pts, key=lambda r: r[4])  # by raw time
        for idx, p in enumerate(pts_sorted, start=1):
            if p[0] == select_id:
                return idx
        return ""

    def delete_selected_reversal_event(self):
        """
        Delete selected reversal events from the reversal table + overlay markers.
        """
        if not self.reversal_tree:
            return
        sel = self.reversal_tree.selection()
        if not sel:
            return
        confirm = messagebox.askokcancel("Confirm Deletion", f"Delete {len(sel)} reversal event(s)?")
        if not confirm:
            return

        keys_to_remove = set()
        for iid in sel:
            # iid like "T2:7"
            try:
                track, select_str = iid.split(":")
                select_id = int(select_str)
                keys_to_remove.add((track, select_id))
            except:
                continue

        # remove from list
        self.reversal_events = [
            ev for ev in self.reversal_events
            if (ev["track"], ev["select_id"]) not in keys_to_remove
        ]

        # remove from marker set
        self.reversal_point_keys -= keys_to_remove

        self.refresh_reversal_table()
        self.update_image()

    def _pick_nearest_point_on_track(self, track_id, click_x, click_y):
        """
        Find the nearest point on a track in image x,y space.
        Preference: points matching current (z, t) if any exist.
        Returns the matched raw point tuple: (select_id, x, y, z, t) or None.
        """
        pts = self.raw_tracks.get(track_id, [])
        if not pts:
            return None

        current_z = parse_z_value(self.subdirs[self.sub_index])
        current_t = self.time_index

        # Prefer points on the current frame (same z and t)
        candidates = [p for p in pts if (p[3] == current_z and p[4] == current_t)]
        if not candidates:
            candidates = pts

        best = None
        best_d2 = None
        for p in candidates:
            dx = p[1] - click_x
            dy = p[2] - click_y
            d2 = dx*dx + dy*dy
            if best is None or d2 < best_d2:
                best = p
                best_d2 = d2

        # enforce a radius tolerance in pixels
        if best is None:
            return None
        if best_d2 is not None and best_d2 > (self.reversal_pick_radius ** 2):
            return None
        return best

    def mark_reversal_event_from_click(self, click_x, click_y):
        """
        In reversal mode: requires exactly one selected track.
        Click near a point -> store that point's x,y,z,t as reversal event.
        """
        # must have exactly one selected track
        if not self.selected_tracks or len(self.selected_tracks) != 1:
            messagebox.showinfo("Select Track", "Please select exactly one track (e.g., T2) before marking a reversal event.")
            return

        track_id = self.selected_tracks[0]
        pt = self._pick_nearest_point_on_track(track_id, click_x, click_y)
        if pt is None:
            messagebox.showinfo("No Point Found", f"No nearby point found on {track_id}. Try clicking closer to an existing point.")
            return

        select_id, x, y, z, t = pt
        key = (track_id, select_id)

        # prevent duplicates
        if key in self.reversal_point_keys:
            # optional: allow toggling off
            remove = messagebox.askyesno("Already Marked", "This point is already marked as a reversal event.\nRemove it?")
            if remove:
                self.reversal_point_keys.remove(key)
                self.reversal_events = [ev for ev in self.reversal_events if (ev["track"], ev["select_id"]) != key]
                self.refresh_reversal_table()
                self.update_image()
            return

        self.reversal_point_keys.add(key)
        self.reversal_events.append({
            "track": track_id,
            "select_id": select_id,
            "x": x,
            "y": y,
            "z": z,
            "t": t
        })

        # ensure window exists/shown in reversal mode
        self.show_reversal_window()
        self.refresh_reversal_table()
        self.update_image()



    def on_translucence_change(self, val_str):
        try:
            percent = float(val_str)
        except ValueError:
            percent = 100
        self.tracks_alpha = percent / 100.0
        self.update_image()


# --- MAIN ---
if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = ImageTrackerApp(root)
    root.mainloop()
