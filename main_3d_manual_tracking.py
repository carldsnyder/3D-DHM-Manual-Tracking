# =============================================================================
# IMPORTS SECTION
# These lines bring in external "libraries" (pre-written collections of tools)
# that the program needs in order to work. Think of them like loading a toolbox
# before you start a job.
# =============================================================================

import os           # os = Operating System tools: used to work with file paths, folders, etc.
import glob         # glob: used to search for files matching a pattern (e.g., all *.png files)
import threading    # threading: allows the program to run tasks in the background without freezing the UI
import shutil       # shutil: used for file operations like copying or deleting whole folders
from datetime import datetime   # datetime: used to get the current date/time (for autosave folder names)

import tkinter as tk                        # tkinter: Python's built-in GUI (Graphical User Interface) library
from tkinter import messagebox              # messagebox: used to show pop-up dialog boxes (errors, confirmations, etc.)
from tkinter import ttk                     # ttk: a newer, styled set of tkinter widgets (e.g., the Treeview table)

from PIL import Image       # PIL/Pillow: a library for opening and manipulating image files
import numpy as np          # numpy: a library for working with large arrays of numbers efficiently
import colorsys             # colorsys: used to convert between color formats (e.g., HSV -> RGB)

# Matplotlib is used to display images and draw overlays (track lines and markers).
import matplotlib
matplotlib.use("TkAgg")     # Tell matplotlib to render inside a Tkinter window (not a separate popup)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg   # Embeds a matplotlib canvas in the tkinter window
from matplotlib.figure import Figure                               # Figure is the top-level container for a matplotlib plot

import tkinter.filedialog as filedialog     # filedialog: used to open the folder-picker dialog boxes


# =============================================================================
# STANDALONE (MODULE-LEVEL) HELPER FUNCTIONS
# These functions are defined outside the main class and can be called
# independently. They perform small, reusable utility tasks.
# =============================================================================

def get_image_files(directory):
    """
    Searches a given folder for image files and returns them as a sorted list.

    It looks for files with these extensions: .png, .tif, .tiff, .jpg, .jpeg, .bmp
    The files are sorted alphabetically/numerically so they appear in the correct order
    (e.g., frame_001.png, frame_002.png, frame_003.png ...).
    """
    # Define the list of image file types to look for
    extensions = ('*.png', '*.tif', '*.tiff', '*.jpg', '*.jpeg', '*.bmp')
    files = []   # Start with an empty list

    # Loop through each extension pattern and find matching files in the directory
    for ext in extensions:
        # glob.glob finds all files matching the pattern in the given directory
        files.extend(glob.glob(os.path.join(directory, ext)))

    # Return the full list sorted so images appear in the right order
    return sorted(files)


def determine_imagedata_type(pil_image):
    """
    Returns the "mode" of an image (e.g., 'L' for grayscale, 'RGB' for color).
    This tells the program how the pixel data is structured so it can display it correctly.
    """
    return pil_image.mode


def parse_z_value(dir_path):
    """
    Reads the name of a folder and converts it to a floating-point (decimal) number,
    which represents the Z-depth (focus plane) for that folder's images.

    Example: a folder named '-17.000' would return -17.0

    If the folder name is not a valid number, it returns None (meaning 'no value').
    This is used to sort and navigate between Z-planes.
    """
    try:
        # os.path.basename extracts just the folder name (not the full path)
        # float(...) converts the string name to a decimal number
        return float(os.path.basename(dir_path))
    except ValueError:
        # If it fails (e.g., folder name is "images" not "-17.000"), return None
        return None


def rgb_to_hex(rgb):
    """
    Converts a color from RGB tuple format (where each channel is a value 0.0–1.0)
    into a hex color string like '#ff5733', which is used by tkinter and matplotlib
    to specify colors.

    Example: (1.0, 0.341, 0.2) → '#ff5733'
    """
    # The f-string formats each color channel: clamp to [0,1], multiply by 255, convert to 2-digit hex
    return '#' + ''.join(f'{int(max(0, min(1, c))*255):02x}' for c in rgb)


def get_track_color(track_index):
    """
    Returns a color (as a hex string) for a given track number.

    - For the first 20 tracks (index 0–19): uses a hand-picked list of 20 visually distinct colors.
    - For tracks 21–50 (index 20–49): automatically generates colors using the "golden ratio" method,
      which spreads colors evenly around the color wheel so they don't look too similar to each other.

    This ensures each track in the overlay has its own unique, identifiable color.
    """
    if track_index < 20:
        # A carefully chosen palette of 20 colors that are easy to distinguish from one another
        distinct_colors = [
            '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
            '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
            '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000',
            '#aaffc3', '#808000', '#ffd8b1', '#000075', '#ff00ff'
        ]
        return distinct_colors[track_index]   # Return the color at the given index
    else:
        # For track index 20 and beyond, generate colors mathematically
        golden_ratio_conjugate = 0.61803398875   # A special number that distributes hues evenly
        hue = ((track_index - 20) * golden_ratio_conjugate) % 1   # Compute a unique hue in [0, 1]
        saturation = 0.8    # Fairly vivid (not washed out)
        value = 0.8         # Not too dark
        # Convert HSV (Hue/Saturation/Value) color model to RGB channels
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert those RGB floats to a hex color string
        return rgb_to_hex((r, g, b))


# =============================================================================
# MAIN APPLICATION CLASS
# Everything below is the main program, organized as a "class" — a self-contained
# blueprint that bundles together all the data and behavior of the application.
# When Python runs `ImageTrackerApp(root)`, it creates one instance of this class,
# which sets up the entire window, loads images, and handles all user interactions.
# =============================================================================

class ImageTrackerApp:
    """
    The main application class for the 3D manual particle/object tracker.

    Key features:
      1. Tracks are named T1, T2, T3, etc.
      2. Left/right arrow keys navigate through the time (image frame) index.
      3. Up/down arrow keys adjust the Z-plane (depth) one step at a time.
      4. A Unit Conversion panel lets the user scale raw pixel/frame values
         into real-world units (e.g., microns, seconds).
      5. Raw (unconverted) track data is always stored internally so that
         unit conversion is only applied at display/save time.
      6. Autosave of tracks every 60 seconds if there are unsaved changes.
      7. Detects duplicate time points when adding a new point.
      8. Computes global image min/max brightness in a background thread so the UI stays responsive.
      9. Saves a 'unitconv.npy' file alongside the track .npy files to record the conversion factors.
    """

    # -------------------------------------------------------------------------
    # __init__: The "constructor" — called once when the app is first created.
    # It sets up all variables, asks the user for folders, builds the UI, and
    # starts background processes.
    # -------------------------------------------------------------------------
    def __init__(self, root):
        self.root = root   # 'root' is the main tkinter window object
        self.root.title("Image Tracker with Binary Contrast at 100%")   # Set the window title

        # --- Prompt the user to choose input and output folders ---
        # filedialog.askdirectory opens a folder-picker dialog box
        self.input_dir = filedialog.askdirectory(title="Select Path to Input Directory")
        self.output_dir = filedialog.askdirectory(title="Select Path to Output Directory")

        # --- Find all valid Z-plane subdirectories inside the input folder ---
        # os.listdir() lists everything in the input folder
        # os.path.isdir() checks if each item is a folder (not a file)
        # parse_z_value() checks if the folder name is a valid number
        all_subdirs = [
            os.path.join(self.input_dir, d)
            for d in os.listdir(self.input_dir)
            if os.path.isdir(os.path.join(self.input_dir, d)) and parse_z_value(os.path.join(self.input_dir, d)) is not None
        ]

        # If no valid Z-plane subfolders were found, show an error and close the app
        if not all_subdirs:
            messagebox.showerror("Error", "No properly named sub-directories found.")
            root.destroy()
            return

        # Sort subdirectories from highest Z-value to lowest (reverse=True)
        # This means index 0 corresponds to the top Z-plane
        self.subdirs = sorted(all_subdirs, key=lambda d: parse_z_value(d), reverse=True)

        # Build a dictionary: each subdir → list of image file paths in that subdir
        # This avoids re-scanning the disk every time the user navigates
        self.images_by_subdir = {sub: get_image_files(sub) for sub in self.subdirs}

        # --- Navigation state variables ---
        self.sub_index = 0    # Which Z-plane (subdir) is currently displayed (index into self.subdirs)
        self.time_index = 0   # Which time frame (image) is currently displayed

        # --- Track data storage ---
        # raw_tracks: a dictionary mapping track ID (e.g., "T1") to a list of points.
        # Each point is stored as a tuple: (select_id, x_raw, y_raw, z_raw, t_raw)
        #   - select_id: a unique integer assigned to each click, used as the row identifier in the table
        #   - x_raw, y_raw: pixel coordinates of the click on the image
        #   - z_raw: the Z-plane value (a float parsed from the folder name)
        #   - t_raw: the time index (integer frame number)
        self.raw_tracks = {}

        self.edit_mode_track = None   # The track currently being edited (only one at a time)
        self.track_order = []         # A list of track IDs in the order they were created (e.g., ["T1", "T2"])
        self.row_counter = 1          # An ever-increasing counter used to assign unique Select# IDs to new points

        # --- Region selection state ---
        self.locked_region = None         # A (t_min, t_max) tuple defining a selected time range in the table
        self.last_clicked_item = None     # The last row clicked in the edit-mode table (used for Shift-click range selection)

        # --- Zoom state ---
        # These store the current visible X/Y range of the image display.
        # None means "fit the full image".
        self.zoom_xlim = None
        self.zoom_ylim = None

        # --- Image display settings ---
        self.contrast_value = 50           # Default position of the contrast slider (50 = no enhancement)
        self.show_tracks_overlay = False   # Whether track lines/markers are drawn over the image
        self.track_name_to_id = {}         # Maps a track name (e.g., "T1") to its internal ID (also "T1")
        self.last_added_point = None       # The most recently added (x, y, z, t) raw point (used for "recent" row highlight)
        self.selected_tracks = []          # List of track IDs currently selected (highlighted) in the All Tracks listbox

        self.last_selection_mode = "overlay"   # Whether the last listbox interaction was "overlay" or "edit" mode
        self.tracks_alpha = 0.5                # Opacity of the track overlay (0.0 = invisible, 1.0 = fully opaque)

        # --- Reversal Event mode state ---
        # The app has two modes: "Tracking" (click to add points) and "Reversal event" (click to mark a reversal)
        self.mode_var = tk.StringVar(value="Tracking")   # StringVar so the dropdown menu can read/write this value
        self.reversal_window = None    # Reference to the secondary "Reversal Events" pop-up window (created on demand)
        self.reversal_tree = None      # Reference to the Treeview table inside the reversal events window

        # reversal_events: a list of dictionaries, each representing one marked reversal event.
        # Each dict holds the RAW (unconverted) coordinates:
        #   {"track": "T2", "select_id": 7, "x": 120.0, "y": 85.0, "z": -17.0, "t": 5}
        self.reversal_events = []

        # reversal_point_keys: a set of (track_id, select_id) tuples for quick lookup.
        # Points in this set are drawn with an 'X' marker instead of a circle on the overlay.
        self.reversal_point_keys = set()

        # How close (in image pixels) a mouse click must be to an existing point
        # to count as "selecting" it for reversal event marking
        self.reversal_pick_radius = 12

        # --- Unit conversion ratios ---
        # These multiply the raw pixel/frame values to convert them into real-world units.
        # Default of 1.0 means "no conversion" (displayed values equal raw values).
        self.x_ratio = 1.0
        self.y_ratio = 1.0
        self.z_ratio = 1.0
        self.t_ratio = 1.0

        # --- Keyboard state ---
        self.e_key_pressed = False    # True while the 'e' key is held down (used for e+click = enter edit mode)
        self.sorted_tracks = []       # A sorted list of track names, kept in sync with self.raw_tracks

        # --- Global brightness range (computed in background) ---
        # These are used for the contrast slider and consistent image display
        self.global_min = None   # The darkest pixel value found across all images
        self.global_max = None   # The brightest pixel value found across all images

        # --- Autosave state ---
        self.autosave_job = None    # Reference to the scheduled autosave timer (so it can be cancelled)
        self.autosave_dir = None    # Path to the most recent autosave folder
        self.dirty = False          # True if there are unsaved changes since the last save

        # --- Build the user interface ---
        self.build_ui()

        # Start computing global min/max pixel values across all images in the background
        # This runs in a separate thread so the UI is not frozen during scanning
        self.precompute_global_min_max_with_progress()

        # --- Bind keyboard shortcuts to the entire application window ---
        # bind_all means the shortcut works regardless of which widget has focus
        self.root.bind_all("<Left>", self.on_left_key)          # Left arrow → go to previous time frame
        self.root.bind_all("<Right>", self.on_right_key)        # Right arrow → go to next time frame
        self.root.bind_all("<Up>", self.on_up_key)              # Up arrow → move to higher Z-plane
        self.root.bind_all("<Down>", self.on_down_key)          # Down arrow → move to lower Z-plane
        self.root.bind("<Delete>", lambda event: self.delete_selection())          # Delete key → delete selected row/track
        self.track_tree.bind("<Delete>", lambda event: self.delete_selection())   # Delete key also works in the data table
        self.canvas.mpl_connect("scroll_event", self.on_scroll)                  # Mouse scroll wheel → zoom image
        self.root.bind("<KeyPress-e>", self.on_e_key_press)       # 'e' key down → set flag
        self.root.bind("<KeyRelease-e>", self.on_e_key_release)   # 'e' key up → clear flag

        # Draw the first image on startup
        self.update_image()


    # =========================================================================
    # AUTOSAVE METHODS
    # These methods handle automatically saving track data in the background
    # every 60 seconds if there have been any changes since the last save.
    # =========================================================================

    def _ensure_autosave_scheduled(self):
        """
        If no autosave timer is currently running, schedule one to fire in 60 seconds.
        'root.after(60000, func)' tells tkinter to call func after 60,000 milliseconds (60 seconds).
        """
        if self.autosave_job is None:
            self.autosave_job = self.root.after(60000, self.autosave_tracks)

    def _converted_points_for_saving(self, raw_points):
        """
        Takes a list of raw track points and returns a numpy array ready to be saved to disk.

        The output array has exactly 5 columns per row:
            (ID#, x, y, z, t)
        where:
          - ID# is assigned by sorting points by their raw time value (smallest time = ID 1)
          - x, y, z, t values are multiplied by the unit conversion ratios and rounded

        Returns None if there are no points to save.
        """
        if not raw_points:
            return None   # Nothing to save

        # Sort the raw points by their time value (index 4 in the tuple)
        sorted_points = sorted(raw_points, key=lambda r: r[4])

        # Build the converted list of (ID#, conv_x, conv_y, conv_z, conv_t)
        converted = []
        for id_num, point in enumerate(sorted_points, start=1):   # id_num starts at 1
            conv_x = round(point[1] * self.x_ratio, 4)   # Multiply pixel X by x conversion ratio
            conv_y = round(point[2] * self.y_ratio, 4)   # Multiply pixel Y by y conversion ratio
            conv_z = round(point[3] * self.z_ratio, 4)   # Multiply Z depth by z conversion ratio
            conv_t = round(point[4] * self.t_ratio, 4)   # Multiply frame index by time ratio
            converted.append((id_num, conv_x, conv_y, conv_z, conv_t))

        # np.array converts the list into a numpy array (a grid of numbers)
        return np.array(converted)

    def _track_num_from_id(self, track_id: str):
        """
        Converts a track name string like 'T2' into just the integer 2.
        This is used when building the reversal events array for saving.
        Returns None if the conversion fails.
        """
        try:
            return int(track_id[1:])   # Remove the 'T' prefix and convert the rest to int
        except:
            return None

    def _reversal_events_array_for_saving(self):
        """
        Builds and returns a numpy array of all reversal events, ready to be saved to disk.

        Output array has 7 columns per row (Nx7):
            track_num, select_id, id_num, x, y, z, t
        where x, y, z, t are in CONVERTED (real-world) units.

        Returns None if there are no reversal events.
        """
        if not self.reversal_events:
            return None   # Nothing to save

        rows = []
        for ev in self.reversal_events:
            track_id = ev.get("track")           # e.g. "T2"
            select_id = int(ev.get("select_id")) # The unique row ID of this point
            track_num = self._track_num_from_id(track_id)   # e.g. 2
            if track_num is None:
                continue   # Skip if the track name can't be parsed

            # Get the temporal ID# for this point (its position when sorted by time)
            id_num = self.get_temporal_id_for_select_id(track_id, select_id)
            if id_num == "":
                id_num = 0   # Default to 0 if it can't be determined

            # Apply unit conversion to coordinates before saving
            conv_x = round(ev["x"] * self.x_ratio, 4)
            conv_y = round(ev["y"] * self.y_ratio, 4)
            conv_z = round(ev["z"] * self.z_ratio, 4)
            conv_t = round(ev["t"] * self.t_ratio, 4)

            rows.append((track_num, select_id, int(id_num), conv_x, conv_y, conv_z, conv_t))

        if not rows:
            return None

        # Convert to numpy array with float64 precision (64-bit floating-point numbers)
        return np.array(rows, dtype=np.float64)

    def _load_reversal_events_npy(self, reversal_dir, x_saved, y_saved, z_saved, t_saved):
        """
        Loads reversal events from a previously saved file:
            reversal_dir/reversal_events.npy

        The file is expected to have 7 columns (Nx7):
            track_num, select_id, id_num, x, y, z, t
        where x, y, z, t are in the CONVERTED units that were used when saving.

        This method converts them back to RAW internal units by dividing by the saved ratios,
        then populates self.reversal_events and self.reversal_point_keys.

        Returns True if loading succeeded, False otherwise.
        """
        path = os.path.join(reversal_dir, "reversal_events.npy")
        if not os.path.isfile(path):
            return False   # File doesn't exist, nothing to load

        try:
            arr = np.load(path)   # Load the numpy array from disk
        except Exception as e:
            messagebox.showwarning("Warning", f"Could not load reversal_events.npy:\n{e}")
            return False

        # Validate the array shape: must be a 2D array with exactly 7 columns
        if arr.ndim != 2 or arr.shape[1] != 7:
            messagebox.showwarning("Warning", "reversal_events.npy has unexpected shape; expected Nx7.")
            return False

        # Clear any existing reversal data before loading
        self.reversal_events = []
        self.reversal_point_keys = set()

        existing_tracks = set(self.raw_tracks.keys())   # Set of loaded track IDs like {"T1", "T2"}

        # Process each row of the saved array
        for row in arr:
            try:
                track_num = int(row[0])      # Column 0: track number (e.g., 2 for T2)
                select_id = int(row[1])      # Column 1: the select_id of the point
                # id_num = int(row[2])       # Column 2: temporal ID# (not needed for restore)
                conv_x = float(row[3])       # Column 3: converted X coordinate
                conv_y = float(row[4])       # Column 4: converted Y coordinate
                conv_z = float(row[5])       # Column 5: converted Z coordinate
                conv_t = float(row[6])       # Column 6: converted time
            except Exception:
                continue   # Skip malformed rows

            track_id = f"T{track_num}"   # Reconstruct track name like "T2"
            if track_id not in existing_tracks:
                continue   # Skip if this track wasn't found during loading

            # Reverse the unit conversion to get back to raw internal units
            raw_x = conv_x / x_saved
            raw_y = conv_y / y_saved
            raw_z = conv_z / z_saved
            raw_t = conv_t / t_saved

            # Verify that this select_id actually exists in the loaded track
            pts = self.raw_tracks.get(track_id, [])
            if not any(p[0] == select_id for p in pts):
                continue   # Cannot reliably map this event; skip it

            # Add the event back into the live reversal data structures
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
        """
        Automatically saves all track data to a timestamped folder in the output directory.
        Only runs if there are unsaved changes (self.dirty == True).

        The save structure is:
            output_dir/
              Autosaved_Manual_Tracks DD_MM_YY HH-MM/
                TrackData/
                  T1.npy, T2.npy, ... (one file per track)
                  unitconv.npy        (the conversion ratios)
                ReversalData/
                  reversal_events.npy (if any reversal events exist)

        After saving, it schedules the next autosave for 60 seconds later.
        """
        if self.dirty:
            # Delete the previous autosave folder (to avoid accumulating many folders)
            if self.autosave_dir and os.path.isdir(self.autosave_dir):
                shutil.rmtree(self.autosave_dir)   # shutil.rmtree deletes a folder and all its contents

            # Create a new folder name using the current date and time
            timestamp = datetime.now().strftime("%d_%m_%y %H-%M")   # e.g., "09_03_26 14-35"
            dir_name = f"Autosaved_Manual_Tracks {timestamp}"
            new_dir = os.path.join(self.output_dir, dir_name)
            os.makedirs(new_dir, exist_ok=True)   # Create the folder (exist_ok=True = don't fail if it already exists)

            # Create TrackData and ReversalData subdirectories inside the autosave folder
            track_dir = os.path.join(new_dir, "TrackData")
            reversal_dir = os.path.join(new_dir, "ReversalData")
            os.makedirs(track_dir, exist_ok=True)
            os.makedirs(reversal_dir, exist_ok=True)

            # Save each track as a .npy file (numpy binary format) in the TrackData folder
            for track_id, raw_points in self.raw_tracks.items():
                arr = self._converted_points_for_saving(raw_points)   # Convert to save format
                if arr is None:
                    continue   # Skip tracks with no points
                np.save(os.path.join(track_dir, f"{track_id}.npy"), arr)   # Save to disk

            # Save the unit conversion ratios so they can be restored when loading
            np.save(
                os.path.join(track_dir, "unitconv.npy"),
                np.array([self.x_ratio, self.y_ratio, self.z_ratio, self.t_ratio])
            )

            # Save reversal events if there are any
            rev_arr = self._reversal_events_array_for_saving()
            if rev_arr is not None:
                np.save(os.path.join(reversal_dir, "reversal_events.npy"), rev_arr)

            self.autosave_dir = new_dir   # Remember this autosave folder path for next time
            self.dirty = False            # Reset the "unsaved changes" flag

        # Schedule the next autosave to run in another 60 seconds
        self.autosave_job = self.root.after(60000, self.autosave_tracks)


    # =========================================================================
    # KEYBOARD EVENT HANDLERS: 'e' KEY
    # The 'e' key is used in combination with a listbox click to switch a track
    # into edit mode. These handlers simply track whether 'e' is currently held.
    # =========================================================================

    def on_e_key_press(self, event):
        """Called when the 'e' key is pressed down. Sets a flag to True."""
        self.e_key_pressed = True

    def on_e_key_release(self, event):
        """Called when the 'e' key is released. Clears the flag back to False."""
        self.e_key_pressed = False


    # =========================================================================
    # UNIT CONVERSION
    # The user can enter multiplier values into text boxes to convert raw
    # pixel/frame values into real-world units (e.g., microns, seconds).
    # =========================================================================

    def update_conversion_factors(self, event=None):
        """
        Reads the values from the four unit conversion entry boxes (X, Y, Z, Time)
        and updates the corresponding ratio variables.

        If the user enters something that's not a valid number, the ratio defaults to 1.0.
        After updating, the track data table is refreshed to show the new converted values.
        The 'event=None' allows this to be called both from button clicks and key bindings.
        """
        # Try to read each entry box and convert to a float; fall back to 1.0 on error
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

        # Refresh the data table to show values with the new conversion applied
        self.update_track_table()

        # Move keyboard focus back to the main window so arrow keys work normally again
        self.root.focus_set()


    # =========================================================================
    # TRACK LISTBOX REFRESH
    # The "All Tracks" listbox on the right side shows a list of all track names.
    # This method rebuilds that list and applies color-coding to show which track
    # is in edit mode (yellow) vs. selected for overlay (blue).
    # =========================================================================

    def refresh_track_listbox(self, preserve_view=False):
        """
        Clears and rebuilds the "All Tracks" listbox from self.raw_tracks.

        If preserve_view=True, the scroll position is remembered and restored
        so the list doesn't jump back to the top when updated.
        """
        if preserve_view:
            # Save the current scroll position (a fraction between 0.0 and 1.0)
            current_yview = self.track_listbox.yview()

        # Sort the track names numerically: T1, T2, T3, ... (not T1, T10, T11, T2, ...)
        self.sorted_tracks = sorted(self.raw_tracks.keys(), key=lambda s: int(s[1:]))
        self.track_order = self.sorted_tracks[:]   # Keep track_order in sync

        # Clear the listbox and re-insert all track names
        self.track_listbox.delete(0, tk.END)
        for i, track in enumerate(self.sorted_tracks):
            self.track_listbox.insert(tk.END, track)   # Add track name to the end of the list

            # Apply background color based on the track's current state:
            if track == self.edit_mode_track and self.last_selection_mode == "edit":
                self.track_listbox.itemconfig(i, {'bg': 'lightyellow'})  # Yellow = currently being edited
            elif track in self.selected_tracks:
                self.track_listbox.itemconfig(i, {'bg': 'lightblue'})    # Blue = selected for overlay display
            else:
                self.track_listbox.itemconfig(i, {'bg': 'white'})        # White = default (not selected)

        if preserve_view:
            # Restore the previous scroll position
            self.track_listbox.yview_moveto(current_yview[0])


    # =========================================================================
    # CONTRAST / BRIGHTNESS LOGIC
    # These methods handle the contrast slider and apply a brightness transform
    # to the image array before displaying it.
    # =========================================================================

    def on_contrast_scroll(self, val_str):
        """
        Called whenever the contrast slider is moved.
        Updates self.contrast_value and redraws the current image.
        val_str is the slider value as a string (tkinter passes it as a string).
        """
        self.contrast_value = int(val_str)
        self.update_image()

    def apply_contrast_transform(self, arr):
        """
        Applies a contrast enhancement to a numpy image array.

        The contrast_value slider goes from 50 (no change) to 100 (full binary).
        - At 50: no change, return the original array.
        - At 100: full binarization — pixels below midpoint become min, above become max.
        - In between: a linear stretching factor is applied around the midpoint.

        Requires self.global_min and self.global_max to be computed (done in background).
        Returns the modified array as float32.
        """
        # If global min/max haven't been computed yet, return the array unchanged
        if self.global_min is None or self.global_max is None:
            return arr

        gmin = self.global_min
        gmax = self.global_max
        mid = (gmin + gmax) / 2.0   # The midpoint brightness value

        # Normalize the contrast slider: 50→0.0 (no contrast), 100→1.0 (full contrast)
        c_norm = (self.contrast_value - 50) / 50.0

        if c_norm <= 0:
            # No contrast enhancement — return array as-is (just cast to float32)
            return arr.astype(np.float32)
        elif c_norm >= 1:
            # Maximum contrast (binary): everything above midpoint → max, everything below → min
            new_arr = arr.astype(np.float32)
            new_arr[new_arr < mid] = gmin    # Dark pixels become the darkest value
            new_arr[new_arr >= mid] = gmax   # Bright pixels become the brightest value
            return new_arr
        else:
            # Partial contrast enhancement: stretch pixel values away from the midpoint
            # A higher factor makes pixels "jump" more aggressively toward min or max
            factor = 1.0 / (1.0 - c_norm + 1e-8)   # 1e-8 prevents division by zero
            arr_f = arr.astype(np.float32)
            new_arr = mid + factor * (arr_f - mid)   # Stretch values around midpoint
            # Clamp values to valid range so nothing goes out of bounds
            new_arr[new_arr < gmin] = gmin
            new_arr[new_arr > gmax] = gmax
            return new_arr

    def precompute_global_min_max_with_progress(self):
        """
        Scans ALL images in ALL Z-plane subdirectories to find the global
        minimum and maximum pixel brightness values.

        This runs in a background thread so the main UI stays responsive.
        A progress bar is shown at the bottom of the image panel while scanning.

        Result is stored in self.global_min and self.global_max.
        """
        # Count total number of images across all subdirs
        total = sum(len(get_image_files(sub)) for sub in self.subdirs)
        if total == 0:
            # No images found — default to 0–255 range
            self.global_min = 0
            self.global_max = 255
            return

        # Create a tkinter DoubleVar (a variable that the progress bar reads)
        self.progress_var = tk.DoubleVar()
        # Create and display a progress bar at the bottom of the image panel
        self.progress_bar = ttk.Progressbar(self.left_frame, variable=self.progress_var, maximum=total)
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        def _background():
            """
            Inner function that runs in the background thread.
            Opens every image, finds its min/max pixel value, and updates the global range.
            """
            gmin, gmax = None, None   # Start with no known range
            count = 0

            for sub in self.subdirs:
                for img_path in get_image_files(sub):
                    try:
                        pil_img = Image.open(img_path)                  # Open the image file
                        arr = np.array(pil_img)                         # Convert to a numpy array of numbers
                        local_min, local_max = arr.min(), arr.max()     # Find this image's min/max

                        # Update the global min/max if this image extends the range
                        if gmin is None or local_min < gmin:
                            gmin = local_min
                        if gmax is None or local_max > gmax:
                            gmax = local_max
                    except Exception as e:
                        print(f"Warning: cannot open {img_path}: {e}")

                    count += 1
                    # Update the progress bar on the main thread (thread-safe via root.after)
                    self.root.after(0, self.progress_var.set, count)

            # Store the final results in the main app object
            self.global_min, self.global_max = gmin, gmax
            print(f"Global range => min={self.global_min}, max={self.global_max}")

            # Hide the progress bar once scanning is complete
            self.root.after(0, self.progress_bar.pack_forget)

        # Start the background function in a new thread
        # daemon=True means the thread will automatically stop if the main app closes
        threading.Thread(target=_background, daemon=True).start()


    # =========================================================================
    # TREEVIEW (DATA TABLE) SELECTION HANDLERS
    # The edit-mode data table (showing points for the active track) supports
    # three types of row selection:
    #   - Normal click: select one row
    #   - Shift+click: select a range of rows
    #   - Ctrl+click: toggle individual rows on/off
    # =========================================================================

    def on_tree_click_normal(self, event):
        """
        Handles a plain left-click on a row in the data table.
        Clears the listbox selection, selects only the clicked row,
        updates the locked time region, refreshes row colors, and
        jumps the image display to the time of the selected point.
        """
        self.track_listbox.selection_clear(0, tk.END)   # Deselect everything in the All Tracks listbox
        region = self.track_tree.identify("region", event.x, event.y)   # Identify what part was clicked
        if region != "cell":
            return   # Only respond to clicks on actual data cells (not headers or empty space)

        item_id = self.track_tree.identify_row(event.y)   # Find which row was clicked
        if not item_id:
            return

        self.track_tree.selection_set(item_id)   # Select just this one row
        self.last_clicked_item = item_id         # Remember this as the "anchor" for shift-clicks
        self.update_locked_region()              # Update the locked time range
        self.update_row_tags()                   # Refresh row background colors
        self.jump_to_time_from_selection()       # Navigate image to this point's time
        return "break"   # "break" tells tkinter to stop processing this event (prevents double-handling)

    def on_tree_click_shift(self, event):
        """
        Handles Shift+click on a row in the data table.
        Selects all rows between the previously clicked row and the current one.
        """
        self.track_listbox.selection_clear(0, tk.END)
        region = self.track_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        item_id = self.track_tree.identify_row(event.y)
        if not item_id:
            return

        children = list(self.track_tree.get_children())   # All row IDs in table order

        # If there's no previous click to anchor from, just select the current row
        if not self.last_clicked_item or (self.last_clicked_item not in children):
            self.track_tree.selection_set(item_id)
            self.last_clicked_item = item_id
        else:
            # Find the index positions of the anchor and current rows
            idx1 = children.index(self.last_clicked_item)
            idx2 = children.index(item_id)
            start, end = sorted([idx1, idx2])         # Make sure start ≤ end
            sel_range = children[start:end+1]         # Slice out all rows in between
            self.track_tree.selection_set(sel_range)  # Select all of them

        self.update_locked_region()
        self.update_row_tags()
        self.jump_to_time_from_selection()
        return "break"

    def on_tree_click_ctrl(self, event):
        """
        Handles Ctrl+click on a row in the data table.
        Toggles the clicked row's selection on or off without affecting other rows.
        """
        self.track_listbox.selection_clear(0, tk.END)
        region = self.track_tree.identify("region", event.x, event.y)
        if region != "cell":
            return

        item_id = self.track_tree.identify_row(event.y)
        if not item_id:
            return

        cur_sel = set(self.track_tree.selection())   # Get the currently selected rows as a set
        if item_id in cur_sel:
            cur_sel.remove(item_id)   # If already selected, deselect it
        else:
            cur_sel.add(item_id)      # If not selected, add it
            self.last_clicked_item = item_id

        self.track_tree.selection_set(list(cur_sel))   # Apply the updated selection
        self.update_locked_region()
        self.update_row_tags()
        self.jump_to_time_from_selection()
        return "break"

    def update_locked_region(self):
        """
        Updates self.locked_region to be a (t_min, t_max) tuple representing
        the time range covered by the currently selected rows in the data table.

        This is used to highlight a range of rows when the user selects multiple points.
        """
        selected = self.track_tree.selection()   # Get all currently selected row IDs
        times = []
        for iid in selected:
            vals = self.track_tree.item(iid, "values")   # Get the column values for this row
            # Columns are: Select#, ID#, X, Y, Z, Time → Time is at index 5
            if len(vals) >= 6:
                try:
                    times.append(float(vals[5]))   # Extract the Time column value
                except:
                    pass

        # Set the locked region to the span of selected times, or None if no rows are selected
        self.locked_region = (min(times), max(times)) if times else None

    def update_row_tags(self):
        """
        Refreshes the visual highlighting of rows in the data table.

        Rows that are currently selected get the 'selected' tag (light blue background).
        Other visual tags (like z_jump coloring) are preserved.
        """
        for iid in self.track_tree.get_children():
            # Get the existing tags for this row (as a set for easy manipulation)
            current_tags = set(self.track_tree.item(iid, 'tags'))
            current_tags.discard('selected')   # Remove any previous 'selected' tag

            # Re-add 'selected' tag if this row is currently selected
            if iid in self.track_tree.selection():
                current_tags.add('selected')

            self.track_tree.item(iid, tags=list(current_tags))   # Apply updated tags

        # Configure what 'selected' looks like: light blue background
        self.track_tree.tag_configure('selected', background='lightblue')

    def jump_to_time_from_selection(self):
        """
        When the user selects a row (or rows) in the data table,
        this method automatically navigates the image display to show
        the time frame corresponding to the earliest selected point.
        """
        selected = self.track_tree.selection()
        if not selected:
            return

        times = []
        for iid in selected:
            vals = self.track_tree.item(iid, "values")
            # Time is at index 5 (columns: Select#, ID#, X, Y, Z, Time)
            if len(vals) >= 6:
                try:
                    times.append(float(vals[5]))
                except:
                    pass

        if times:
            # Jump to the earliest time in the selection
            self.time_index = int(min(times))
            self.t_scroll.set(self.time_index)   # Update the time slider to match
            self.update_image()                  # Redraw the image at this time


    # =========================================================================
    # Z-PLANE AND TIME NAVIGATION
    # These methods handle moving between Z-planes (depth slices) and time frames
    # via buttons and scrollbars.
    # =========================================================================

    def z_prev(self, event=None):
        """
        Moves one step toward the beginning of the Z-plane list (decreases sub_index).
        Called by the ◄ button next to the Z slider.
        """
        if self.sub_index > 0:
            self.sub_index -= 1
            self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))   # Update slider position
            self.update_image()

    def z_next(self, event=None):
        """
        Moves one step toward the end of the Z-plane list (increases sub_index).
        Called by the ► button next to the Z slider.
        """
        if self.sub_index < len(self.subdirs) - 1:
            self.sub_index += 1
            self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))
            self.update_image()

    def t_prev(self):
        """
        Moves one time frame backward (decreases time_index by 1).
        Called by the ◄ button next to the Time slider.
        """
        if self.time_index > 0:
            new_val = self.time_index - 1
            self.t_scroll.set(new_val)
            self.on_t_scroll(new_val)

    def t_next(self):
        """
        Moves one time frame forward (increases time_index by 1), up to the max frame.
        Called by the ► button next to the Time slider.
        """
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)   # Maximum valid time index
        if self.time_index < max_t:
            new_val = self.time_index + 1
            self.t_scroll.set(new_val)
            self.on_t_scroll(new_val)

    def find_closest_subdir(self, desired, direction="up"):
        """
        Given a desired Z value (e.g., -15.0) and a direction ("up" or "down"),
        finds the index of the subdirectory whose Z-value best matches.

        Direction logic:
          - "up":   prefer subdirs with Z >= desired (moving up in Z)
          - "down": prefer subdirs with Z <= desired (moving down in Z)
        If no match in the desired direction, falls back to the nearest in absolute terms.

        Returns a tuple: (index_into_self.subdirs, actual_z_value)
        """
        best_index = None
        best_diff = float('inf')   # Start with "infinitely far away"

        for i, subdir in enumerate(self.subdirs):
            z_val = parse_z_value(subdir)
            if z_val is None:
                continue   # Skip non-numeric folder names

            # Check if this subdir is in the requested direction
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

        # If no match was found in the requested direction, fall back to any nearest
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
        """
        Called when the Z-plane slider is moved.
        Finds the nearest valid Z-plane to the requested value, updates the display,
        and adjusts the time slider's maximum if the new Z-plane has fewer frames.
        """
        try:
            desired = float(value)   # Convert the slider string value to a float
        except ValueError:
            return

        # Find the closest valid subdir
        closest_index, actual_z = self.find_closest_subdir(desired, direction)
        self.sub_index = closest_index
        self.z_scroll.set(actual_z)                          # Snap slider to the actual Z value
        self.z_value_label.config(text=f"Z: {actual_z}")    # Update the Z value label

        # Clamp time index to the number of frames in this Z-plane
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        if self.time_index > max_t:
            self.time_index = max_t
        self.t_scroll.config(to=max_t)       # Adjust the time slider's maximum
        self.t_scroll.set(self.time_index)

        self.update_image()

    def on_t_scroll(self, value):
        """
        Called when the time slider is moved.
        Updates the current time index and redraws the image.
        """
        self.time_index = int(value)
        self.update_image()


    # =========================================================================
    # KEYBOARD NAVIGATION HANDLERS
    # Arrow keys control navigation:
    #   Left/Right → time frame (earlier/later)
    #   Up/Down → Z-plane (higher/lower depth)
    # =========================================================================

    def on_left_key(self, event):
        """Go to the previous time frame (Left arrow key)."""
        if self.time_index > 0:
            self.time_index -= 1
            self.t_scroll.set(self.time_index)
            self.update_image()

    def on_right_key(self, event):
        """Go to the next time frame (Right arrow key)."""
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        if self.time_index < max_t:
            self.time_index += 1
            self.t_scroll.set(self.time_index)
            self.update_image()

    def on_up_key(self, event):
        """
        Move up one Z-plane step (Up arrow key).
        The Z slider moves up by 1 unit, and find_closest_subdir locates the nearest plane.
        Returns "break" to prevent the Up key from scrolling the listbox.
        """
        current_z = float(self.z_scroll.get())
        max_z = float(self.z_scroll.cget("to"))   # The maximum value the Z slider can reach
        if current_z < max_z:
            new_z = current_z + 1
            self.z_scroll.set(new_z)
            self.on_z_scroll(new_z, direction="up")
        return "break"

    def on_down_key(self, event):
        """
        Move down one Z-plane step (Down arrow key).
        Returns "break" to prevent the Down key from scrolling the listbox.
        """
        current_z = float(self.z_scroll.get())
        min_z = float(self.z_scroll.cget("from"))   # The minimum value the Z slider can reach
        if current_z > min_z:
            new_z = current_z - 1
            self.z_scroll.set(new_z)
            self.on_z_scroll(new_z, direction="down")
        return "break"


    # =========================================================================
    # IMAGE DISPLAY
    # These methods handle loading the correct image from disk, applying
    # contrast enhancement, drawing track overlays, and rendering everything
    # into the matplotlib canvas.
    # =========================================================================

    def update_image(self):
        """
        The main display method — called whenever the image needs to be refreshed.

        Steps:
        1. Determine which image file to load based on current Z-plane and time index.
        2. Open the image and convert it to a numpy array.
        3. Apply the contrast transformation.
        4. Display the image in the matplotlib axes.
        5. Restore any zoom level the user has set.
        6. If track overlay is enabled, draw all track lines and markers.
        7. Redraw the canvas.
        """
        c_subdir = self.subdirs[self.sub_index]   # Current Z-plane folder
        imgs = self.images_by_subdir[c_subdir]    # List of image paths in this folder

        # If no images exist in this Z-plane, show a placeholder message
        if not imgs:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "No images in this directory", ha="center", va="center", color="white")
            self.canvas.draw()
            self.current_image_array = None
            return

        # Clamp time_index to the valid range
        if self.time_index >= len(imgs):
            self.time_index = len(imgs) - 1

        img_path = imgs[self.time_index]   # Get the path to the current image file

        # Try to open the image file; show an error dialog if it fails
        try:
            pil_img = Image.open(img_path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open {img_path}\n{e}")
            return

        # Convert the PIL image to a numpy array of float32 numbers for processing
        arr = np.array(pil_img, dtype=np.float32)
        self.current_image_array = arr   # Store for pixel value inspection on mouse hover

        self.ax.clear()   # Clear the previous image from the plot

        # Apply the contrast transformation
        arr_c = self.apply_contrast_transform(arr)

        # Set the brightness display range based on global min/max (for consistent contrast)
        vmin = self.global_min if self.global_min is not None else arr_c.min()
        vmax = self.global_max if self.global_max is not None else arr_c.max()

        # Display the image:
        # - Grayscale images (2D array) use a gray colormap
        # - Color images (3D array with RGB/RGBA) are displayed in color
        if len(arr_c.shape) == 2:
            self.ax.imshow(arr_c, cmap="gray", vmin=vmin, vmax=vmax)
        else:
            mode = determine_imagedata_type(pil_img)
            if mode in ["RGB", "RGBA"]:
                self.ax.imshow(arr_c.astype(np.uint8))   # Color image: cast back to 8-bit for display
            else:
                self.ax.imshow(arr_c, cmap="gray", vmin=vmin, vmax=vmax)

        # Restore any zoom that the user has set (so zooming persists across frame changes)
        if self.zoom_xlim is not None and self.zoom_ylim is not None:
            self.ax.set_xlim(self.zoom_xlim)
            self.ax.set_ylim(self.zoom_ylim)
        else:
            # First time displaying: record the default zoom level (full image)
            self.zoom_xlim = self.ax.get_xlim()
            self.zoom_ylim = self.ax.get_ylim()

        # If the "Show Tracks" overlay is active, draw all track lines and markers
        if self.show_tracks_overlay:
            self.draw_all_tracks()

        self.ax.axis("off")    # Hide the axis ticks and labels (cleaner display)
        self.canvas.draw()     # Trigger matplotlib to render everything to screen

    def on_canvas_motion(self, event):
        """
        Called whenever the mouse moves over the image canvas.
        Reads the pixel value at the mouse position and displays it in the status label.
        """
        # If no image is loaded, or the mouse is outside the image area, show N/A
        if self.current_image_array is None or event.xdata is None or event.ydata is None:
            self.pixel_info_label.config(text="Pixel: N/A")
            return

        # Convert matplotlib data coordinates to integer pixel indices
        x = int(event.xdata)
        y = int(event.ydata)
        h, w = self.current_image_array.shape[:2]   # Image height and width

        # Make sure the coordinates are within the image bounds
        if not (0 <= x < w and 0 <= y < h):
            self.pixel_info_label.config(text="Pixel: N/A")
            return

        # Read the pixel value at this position (note: array is indexed [row, column] = [y, x])
        val = self.current_image_array[y, x]
        self.pixel_info_label.config(text=f"Pixel:({x},{y}) Value:{val}")

    def on_canvas_click(self, event):
        """
        Called whenever the user clicks on the image canvas.
        Behavior depends on the current mode:

        - In "Reversal event" mode: clicking near an existing track point marks it as a reversal event.
        - In "Tracking" mode: clicking adds a new data point to the active edit-mode track.

        event.xdata and event.ydata are the image coordinates of the click
        (in the image's pixel coordinate system).
        """
        # Ignore clicks outside the image area (e.g., on padding)
        if event.xdata is None or event.ydata is None:
            return

        click_x = int(event.xdata)   # Convert to integer pixel coordinates
        click_y = int(event.ydata)

        # --- Reversal Event Mode ---
        if self.mode_var.get() == "Reversal event":
            self.mark_reversal_event_from_click(click_x, click_y)
            return

        # --- Tracking Mode (normal behavior) ---
        # If no track is in edit mode, and no tracks exist yet, automatically start a new one
        if self.edit_mode_track is None:
            if not self.raw_tracks:
                self.start_new_track()
            else:
                return   # A track exists but none is in edit mode — do nothing

        # Get current position values to form the new data point
        x = click_x
        y = click_y
        z = parse_z_value(self.subdirs[self.sub_index])   # Z value from current folder name
        t = self.time_index                               # Current time frame index

        track_pts = self.raw_tracks[self.edit_mode_track]   # Get the active track's point list

        # Check if a point already exists at this exact time
        existing = [pt for pt in track_pts if pt[4] == t]
        if existing:
            # Ask the user whether to replace the existing point
            replace = messagebox.askyesno(
                "Duplicate Time",
                f"Time point {t} already exists. Replace old data point?"
            )
            if replace:
                # Remove all existing points at this time before adding the new one
                track_pts[:] = [pt for pt in track_pts if pt[4] != t]
            else:
                return   # User chose not to replace — do nothing

        # Create the new raw point tuple: (Select#, x, y, z, t)
        raw_point = (self.row_counter, x, y, z, t)
        track_pts.append(raw_point)                      # Add to the track
        self.last_added_point = (x, y, z, t)            # Mark it for "recent" highlighting in the table
        self.row_counter += 1                            # Increment the global ID counter

        self.update_track_table()   # Refresh the data table to show the new point
        self.jump_to_next_time()    # Automatically advance to the next time frame

        # Mark that there are unsaved changes and ensure autosave is scheduled
        self.dirty = True
        self._ensure_autosave_scheduled()

    def jump_to_next_time(self):
        """
        After adding a new point, automatically advances the display to the next time frame.
        This speeds up the tracking workflow — the user can click through frames without
        manually pressing the forward button each time.
        Stops at the last frame if already at the end.
        """
        c_subdir = self.subdirs[self.sub_index]
        max_t = max(len(self.images_by_subdir[c_subdir]) - 1, 0)
        new_t = self.time_index + 1
        if new_t > max_t:
            new_t = max_t   # Don't go past the last frame
        self.time_index = new_t
        self.t_scroll.set(new_t)
        self.update_image()


    # =========================================================================
    # ZOOM HANDLING (via scroll wheel + Ctrl)
    # The user can zoom in/out on the image using Ctrl+scroll.
    # Zoom is centered on the cursor position.
    # =========================================================================

    def on_scroll(self, event):
        """
        Handles scroll wheel events from matplotlib's event system.
        Only zooms when the Ctrl key is held (event.key contains "control").
        Scroll up = zoom in (make image appear larger).
        Scroll down = zoom out (make image appear smaller).
        """
        # Only zoom if Ctrl is held down
        if event.key is None or "control" not in event.key.lower():
            return

        base_scale = 1.2   # How much to zoom per scroll step (20% change)

        if event.button == 'up':
            scale_factor = 1 / base_scale   # Zoom in: make the visible area smaller
        elif event.button == 'down':
            scale_factor = base_scale        # Zoom out: make the visible area larger
        else:
            scale_factor = 1                # No change

        # Get the current visible X and Y ranges
        cur_xlim = self.zoom_xlim if self.zoom_xlim is not None else self.ax.get_xlim()
        cur_ylim = self.zoom_ylim if self.zoom_ylim is not None else self.ax.get_ylim()

        # Ignore if the cursor is outside the image
        if event.xdata is None or event.ydata is None:
            return

        xdata = event.xdata   # X coordinate of the cursor (in image pixel space)
        ydata = event.ydata   # Y coordinate of the cursor

        # Compute new axis limits centered on the cursor position
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        new_xmin = xdata - (xdata - cur_xlim[0]) * scale_factor
        new_xmax = new_xmin + new_width
        new_ymin = ydata - (ydata - cur_ylim[0]) * scale_factor
        new_ymax = new_ymin + new_height

        # Save and apply the new zoom limits
        self.zoom_xlim = (new_xmin, new_xmax)
        self.zoom_ylim = (new_ymin, new_ymax)
        self.ax.set_xlim(self.zoom_xlim)
        self.ax.set_ylim(self.zoom_ylim)
        self.canvas.draw()

    def on_ctrl_mousewheel(self, event):
        """
        An alternative scroll handler registered via tkinter's event system (for Windows/Mac).
        Works the same as on_scroll but uses tkinter's event.delta instead of button direction.
        event.delta > 0 means scroll up (zoom in); event.delta < 0 means scroll down (zoom out).
        """
        base_scale = 1.2
        if event.delta > 0:
            scale_factor = 1 / base_scale   # Zoom in
        else:
            scale_factor = base_scale        # Zoom out

        # Convert tkinter window pixel coordinates to matplotlib data coordinates
        widget = self.canvas.get_tk_widget()
        height = widget.winfo_height()
        x = event.x
        y = height - event.y   # Flip y-axis: tkinter counts from top, matplotlib from bottom
        try:
            xdata, ydata = self.ax.transData.inverted().transform((x, y))
        except Exception:
            return "break"

        # Compute and apply new zoom limits (same logic as on_scroll)
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
        """
        Resets the zoom level to show the full image (the 'Home' button).
        Sets zoom limits to None so update_image will fit the whole image.
        """
        self.zoom_xlim = None
        self.zoom_ylim = None
        self.update_image()


    # =========================================================================
    # TRACK DATA TABLE (EDIT MODE)
    # When a track is in edit mode, its points are displayed in a sortable table
    # showing Select#, ID#, X, Y, Z, and Time.
    # =========================================================================

    def update_track_table(self):
        """
        Clears and rebuilds the edit-mode data table for the currently active track.

        Each row represents one recorded point in the track.
        Columns: Select# | ID# | X | Y | Z | Time (all converted by unit ratios)

        Visual highlighting:
          - Light yellow, bold = the most recently added point
          - Orange = moderate Z-jump from adjacent points (> 3 units)
          - Red = large Z-jump from adjacent points (> 5 units)
          - Light gray = part of the currently locked (selected) time region
        """
        if self.edit_mode_track is None:
            return   # No track is in edit mode; nothing to display

        raw_points = self.raw_tracks[self.edit_mode_track]
        # Sort points by their raw time value so the table shows them in temporal order
        sorted_points = sorted(raw_points, key=lambda r: r[4])

        # Clear all existing rows from the table
        for iid in self.track_tree.get_children():
            self.track_tree.delete(iid)

        # Pre-compute converted Z values for all points (used for z-jump detection)
        conv_z_list = [round(pt[3] * self.z_ratio, 1) for pt in sorted_points]

        recent_item_id = None   # Will be set if we find the most recently added point

        # Insert one row per point
        for idx, point in enumerate(sorted_points):
            select_id = point[0]        # The unique Select# ID of this point
            id_num = idx + 1            # Temporal ID: 1 = earliest, 2 = second earliest, etc.

            # Apply unit conversion to display values
            conv_x = round(point[1] * self.x_ratio, 1)     # 1 decimal place for X
            conv_y = round(point[2] * self.y_ratio, 1)     # 1 decimal place for Y
            conv_z = conv_z_list[idx]                        # 1 decimal place for Z
            conv_t = round(point[4] * self.t_ratio, 3)     # 3 decimal places for Time

            # --- Compute Z-jump tags ---
            # A large Z jump between consecutive points suggests a potential tracking error
            diff_prev = abs(conv_z - conv_z_list[idx - 1]) if idx > 0 else 0         # Z diff from previous row
            diff_next = abs(conv_z - conv_z_list[idx + 1]) if idx < len(conv_z_list) - 1 else 0  # Z diff from next row
            max_diff = max(diff_prev, diff_next)   # Use the larger of the two differences

            tags = []   # List of tag names to apply to this row
            if max_diff > 5:
                tags.append("z_jump_high")   # Large Z-jump: red background
            elif max_diff > 3:
                tags.append("z_jump_med")    # Moderate Z-jump: orange background

            # --- Selected region tag ---
            # If a time range is locked (from a table row selection), highlight rows within it
            if self.locked_region:
                t_min, t_max = self.locked_region
                if t_min <= conv_t <= t_max:
                    tags.append("selected")

            # --- Recently added tag ---
            # The most recently clicked point gets a bold, yellow highlight
            if self.last_added_point and (point[1], point[2], point[3], point[4]) == self.last_added_point:
                tags.append("recent")

            # Format the row data tuple for display
            disp_point = (select_id, id_num, conv_x, conv_y, conv_z, f"{conv_t:.3f}")

            # Insert this row into the table
            item_id = self.track_tree.insert(
                "", tk.END,
                iid=str(select_id),   # Use Select# as the row's unique identifier (used for deletion)
                values=disp_point,
                tags=tags
            )

            if "recent" in tags:
                recent_item_id = item_id   # Remember this row so we can scroll to it

        # --- Configure tag visual styles ---
        self.track_tree.tag_configure("selected", background="lightgray")
        self.track_tree.tag_configure("recent", font=("TkDefaultFont", 9, "bold"), background="lightyellow")
        self.track_tree.tag_configure("z_jump_med", background="orange")
        self.track_tree.tag_configure("z_jump_high", background="red")

        # --- Auto-scroll to the most recently added row ---
        if recent_item_id:
            children = self.track_tree.get_children()
            total_rows = len(children)
            try:
                # Estimate how many rows are visible in the table widget
                first_bbox = self.track_tree.bbox(children[0])
                visible_count = (self.track_tree.winfo_height() // first_bbox[3]) if first_bbox and first_bbox[3] > 0 else 10
            except Exception:
                visible_count = 10

            recent_index = children.index(recent_item_id)
            if total_rows > visible_count:
                # Scroll so the recent row is centered vertically in the visible area
                fraction = max(0, min((recent_index - (visible_count / 2)) / (total_rows - visible_count), 1))
                self.track_tree.yview_moveto(fraction)
            else:
                self.track_tree.see(recent_item_id)   # Simply ensure it's visible


    # =========================================================================
    # TRACK CREATION AND SAVING
    # =========================================================================

    def start_new_track(self):
        """
        Creates a new empty track with the next available name (T1, T2, T3, ...).
        Sets this new track as the active edit-mode track.
        Clears the data table and resets the row counter for this session.
        """
        # Find the next available track number (e.g., if T1 and T2 exist, use T3)
        n = 1
        while f"T{n}" in self.raw_tracks:
            n += 1
        track_id = f"T{n}"

        # Register the new track
        self.raw_tracks[track_id] = []           # Empty list of points
        self.edit_mode_track = track_id          # Set as active edit track
        self.track_order.append(track_id)        # Add to ordered list
        self.track_info_label.config(text=f"Edit Mode Track: {track_id}")   # Update UI label

        # Clear the data table
        for iid in self.track_tree.get_children():
            self.track_tree.delete(iid)

        self.locked_region = None        # Clear any time selection
        self.row_counter = 1             # Reset the Select# counter for this track
        self.track_name_to_id[track_id] = track_id   # Add to name→id mapping
        self.refresh_track_listbox()     # Refresh the All Tracks list

    def save_tracks(self):
        """
        Manually saves all tracks (and reversal events) to the output directory.

        Saves to:
            output_dir/
              TrackData/
                T1.npy, T2.npy, ...
                unitconv.npy
              ReversalData/
                reversal_events.npy (if any)

        Cancels any pending autosave and deletes the most recent autosave folder
        (since the user has now saved manually).
        """
        if not self.raw_tracks:
            return   # Nothing to save

        # Cancel the autosave timer to avoid double-saving
        if self.autosave_job:
            self.root.after_cancel(self.autosave_job)
            self.autosave_job = None

        # Delete the most recent autosave directory (now superseded by manual save)
        if self.autosave_dir and os.path.isdir(self.autosave_dir):
            shutil.rmtree(self.autosave_dir)
            self.autosave_dir = None

        # Create the output subdirectories
        track_dir = os.path.join(self.output_dir, "TrackData")
        reversal_dir = os.path.join(self.output_dir, "ReversalData")
        os.makedirs(track_dir, exist_ok=True)
        os.makedirs(reversal_dir, exist_ok=True)

        # Save each track as a .npy file
        for track_id, raw_points in self.raw_tracks.items():
            arr = self._converted_points_for_saving(raw_points)   # Convert to save format
            if arr is None:
                continue
            out_path = os.path.join(track_dir, f"{track_id}.npy")
            try:
                np.save(out_path, arr)
                print(f"Saved track {track_id} => {out_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save {track_id}:\n{e}")

        # Save the unit conversion ratios
        unit_path = os.path.join(track_dir, "unitconv.npy")
        try:
            np.save(unit_path, np.array([self.x_ratio, self.y_ratio, self.z_ratio, self.t_ratio]))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save unit conversion:\n{e}")

        # Save reversal events if there are any
        rev_arr = self._reversal_events_array_for_saving()
        if rev_arr is not None:
            rev_path = os.path.join(reversal_dir, "reversal_events.npy")
            try:
                np.save(rev_path, rev_arr)
                print(f"Saved reversal events => {rev_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save reversal events:\n{e}")

        self.dirty = False   # Mark as clean (no unsaved changes)


    # =========================================================================
    # LISTBOX INTERACTION HANDLERS
    # The "All Tracks" listbox on the right lets the user select tracks
    # for overlay display or to enter/exit edit mode.
    # =========================================================================

    def on_listbox_click(self, event):
        """
        Handles a left-click on the All Tracks listbox.

        If the 'e' key is held while clicking:
          → Sets that track as the active edit-mode track (yellow highlight).
        Otherwise:
          → Normal selection behavior (handled by on_select_track).
        """
        index = self.track_listbox.nearest(event.y)   # Find which track was clicked
        if index < 0 or index >= len(self.sorted_tracks):
            return

        track_id = self.sorted_tracks[index]

        if self.e_key_pressed:
            # e + click = enter edit mode for this track
            self.last_selection_mode = "edit"
            self.edit_mode_track = track_id
            self.selected_tracks = [track_id]
            self.track_info_label.config(text=f"Edit Mode Track: {track_id}")
            self.refresh_track_listbox(preserve_view=True)
            self.update_track_table()   # Show this track's points in the data table
            return "break"

    def on_select_track(self, event):
        """
        Called whenever the selection in the All Tracks listbox changes.
        Updates self.selected_tracks and refreshes the image overlay.

        In reversal mode: forces single-track selection (only one track can be active at a time).
        """
        if not self.e_key_pressed:
            self.last_selection_mode = "overlay"   # Regular selection = overlay mode

        selection = self.track_listbox.curselection()   # Get the indices of all selected items
        if not selection:
            self.selected_tracks = []
            return

        chosen = [self.track_listbox.get(i) for i in selection]   # Get the track names

        # In reversal mode, only one track may be selected at a time
        if self.mode_var.get() == "Reversal event" and len(chosen) > 1:
            chosen = [chosen[-1]]   # Keep only the last selected track
            self.track_listbox.selection_clear(0, tk.END)
            # Re-select just that one track in the listbox
            last_track = chosen[0]
            try:
                idx = self.sorted_tracks.index(last_track)
                self.track_listbox.selection_set(idx)
            except:
                pass

        self.selected_tracks = chosen
        self.update_image()   # Redraw with updated overlay highlighting

    def on_listbox_ctrl_click(self, event):
        """
        Handles Ctrl+click on the All Tracks listbox.
        Toggles individual tracks in/out of the selection (multi-select behavior).
        """
        index = self.track_listbox.nearest(event.y)
        if index < 0 or index >= len(self.sorted_tracks):
            return

        track_id = self.sorted_tracks[index]

        # Toggle: if already selected, deselect; if not, select
        if track_id in self.selected_tracks:
            self.track_listbox.selection_clear(index)
        else:
            self.track_listbox.selection_set(index)

        # Rebuild selected_tracks from the current listbox selection
        self.selected_tracks = [self.track_listbox.get(i) for i in self.track_listbox.curselection()]
        self.refresh_track_listbox(preserve_view=True)
        self.update_image()
        return "break"


    # =========================================================================
    # TRACK OVERLAY DRAWING
    # When "Show Tracks" is active, all tracks (or selected ones) are drawn
    # over the image as colored lines and markers.
    # =========================================================================

    def toggle_show_tracks(self):
        """
        Toggles the track overlay on or off.
        Updates the button label to reflect the current state.
        """
        self.show_tracks_overlay = not self.show_tracks_overlay
        self.show_tracks_button.config(text="Hide Tracks" if self.show_tracks_overlay else "Show Tracks")
        self.update_image()   # Redraw to add or remove the overlay

    def draw_all_tracks(self):
        """
        Draws all tracks as lines and point markers over the current image.

        For each track:
          1. A line is drawn connecting all points in temporal order (using their x, y coordinates).
          2. Each point is drawn as a circle marker (open, just the outline).
          3. Points that have been marked as reversal events are drawn as 'X' markers instead.

        Selected tracks (highlighted in the listbox) get thicker lines and markers.
        Track opacity is controlled by self.tracks_alpha.
        """
        for i, track_id in enumerate(self.track_order):
            pts = self.raw_tracks.get(track_id, [])
            if not pts:
                continue   # Skip empty tracks

            # Sort points by raw time so the line connects them in temporal order
            pts_sorted = sorted(pts, key=lambda r: r[4])
            color = get_track_color(i)   # Get a unique color for this track

            # Selected tracks get thicker lines
            if track_id in self.selected_tracks:
                linewidth = 3
                markeredgewidth = 3
            else:
                linewidth = 1
                markeredgewidth = 1

            # Extract X and Y coordinates (raw pixel positions) for drawing
            xvals = [p[1] for p in pts_sorted]
            yvals = [p[2] for p in pts_sorted]

            # 1) Draw the connecting line between all points
            self.ax.plot(
                xvals, yvals,
                linewidth=linewidth,
                color=color,
                linestyle='-',
                alpha=self.tracks_alpha   # Apply opacity setting
            )

            # 2) Split points into normal (circle) and reversal (X) groups
            normal_x, normal_y = [], []   # Points drawn as open circles
            rev_x, rev_y = [], []         # Reversal event points drawn as 'X'

            for p in pts_sorted:
                select_id = p[0]
                key = (track_id, select_id)
                if key in self.reversal_point_keys:
                    rev_x.append(p[1])
                    rev_y.append(p[2])
                else:
                    normal_x.append(p[1])
                    normal_y.append(p[2])

            # Draw normal points as open circles (facecolors='none' = hollow interior)
            if normal_x:
                self.ax.scatter(
                    normal_x, normal_y,
                    marker='o',
                    facecolors='none',      # Hollow circle
                    edgecolors=color,
                    linewidths=markeredgewidth,
                    alpha=self.tracks_alpha
                )

            # Draw reversal event points as 'X' markers (solid, slightly thicker)
            if rev_x:
                self.ax.scatter(
                    rev_x, rev_y,
                    marker='x',
                    c=color,
                    linewidths=markeredgewidth + 1,
                    alpha=self.tracks_alpha
                )


    # =========================================================================
    # LOADING PREVIOUSLY SAVED TRACKS
    # =========================================================================

    def load_previous_tracks(self):
        """
        Prompts the user to select a folder containing previously saved .npy track files,
        then loads all tracks and reversal events from that folder.

        Supports both the new folder structure (TrackData/ + ReversalData/ subdirs)
        and the old flat structure (all .npy files directly in the chosen folder).

        When loading:
        - The unitconv.npy file is read to know what unit conversion was used when saving.
        - Each track .npy file is loaded, its coordinates are converted back to raw units
          by dividing by the saved ratios, and points are given fresh Select# IDs.
        - If track names conflict with already-loaded tracks, conflicting tracks are renamed.
        """
        # Open a folder picker dialog
        load_dir = filedialog.askdirectory(title="Select directory containing NPY track files")
        if not load_dir:
            return   # User cancelled

        # Look for TrackData/ subdirectory (new structure)
        track_dir = os.path.join(load_dir, "TrackData")
        parent_dir = os.path.dirname(load_dir)
        reversal_dir = os.path.join(parent_dir, "ReversalData")

        # Prefer loading from TrackData/ if it exists, otherwise use flat structure
        if os.path.isdir(track_dir):
            npy_files = glob.glob(os.path.join(track_dir, "*.npy"))
        else:
            npy_files = glob.glob(os.path.join(load_dir, "*.npy"))

        if not npy_files:
            return   # No .npy files found

        # --- Load the unit conversion ratios from unitconv.npy ---
        # Default to 1.0 (no conversion) if the file doesn't exist
        x_saved = y_saved = z_saved = t_saved = 1.0
        for fname in ("unitconv.npy", "unitconvert.npy"):   # Try both possible names
            if os.path.isdir(track_dir):
                p = os.path.join(track_dir, fname)
            else:
                p = os.path.join(load_dir, fname)

            if os.path.isfile(p):
                try:
                    x_saved, y_saved, z_saved, t_saved = np.load(p)   # Load the 4 ratios
                except Exception as e:
                    messagebox.showwarning("Warning", f"Could not load unit conversion file {fname}:\n{e}")
                break   # Stop after finding the first valid conversion file

        # Remove the unit conversion file(s) from the list of track files to load
        npy_files = [
            f for f in npy_files
            if os.path.basename(f).lower() not in ("unitconv.npy", "unitconvert.npy")
        ]
        if not npy_files:
            return

        # --- Load each .npy file as a track ---
        loaded_tracks = {}
        for file in npy_files:
            track_name = os.path.splitext(os.path.basename(file))[0]   # e.g., "T1" from "T1.npy"
            try:
                data = np.load(file)   # Load the array from disk

                # Validate: must be a 2D array with exactly 5 columns (ID#, x, y, z, t)
                if data.ndim != 2 or data.shape[1] != 5:
                    messagebox.showerror("Error", f"{file} does not look like a 5-column track file.")
                    continue

                # Convert saved (converted) coordinates back to raw internal units
                # The saved file format is: (ID#, x, y, z, t) with converted values
                temp_points = []
                for _id, x, y, z, t in data:
                    temp_points.append((
                        0,          # Placeholder Select# — will be reassigned below
                        x / x_saved,  # Reverse X conversion
                        y / y_saved,  # Reverse Y conversion
                        z / z_saved,  # Reverse Z conversion
                        t / t_saved   # Reverse time conversion
                    ))

                # Sort by raw time and assign fresh sequential Select# IDs (1, 2, 3, ...)
                temp_points_sorted = sorted(temp_points, key=lambda r: r[4])
                rebuilt = []
                for select_id, pt in enumerate(temp_points_sorted, start=1):
                    rebuilt.append((select_id, pt[1], pt[2], pt[3], pt[4]))

                loaded_tracks[track_name] = rebuilt

            except Exception as e:
                messagebox.showerror("Error", f"Failed to load {file}:\n{e}")

        # Helper to extract the numeric part from a track name like "T3" → 3
        def extract_number(name):
            try:
                return int(name[1:])
            except:
                return None

        # --- Handle naming conflicts ---
        # If a loaded track name already exists in the current session, rename the existing one
        conflicts = []
        for track_name in loaded_tracks.keys():
            if track_name in self.track_name_to_id:
                conflicts.append(track_name)

        # Find the highest track number currently in use (for renaming conflicting tracks)
        max_num = 0
        for name in list(self.track_name_to_id.keys()) + list(loaded_tracks.keys()):
            num = extract_number(name)
            if num and num > max_num:
                max_num = num

        # Rename conflicting existing tracks to avoid clashes with loaded tracks
        for conflict in conflicts:
            track_id = self.track_name_to_id[conflict]
            max_num += 1
            new_name = f"T{max_num}"
            del self.track_name_to_id[conflict]
            self.track_name_to_id[new_name] = track_id

        # --- Merge loaded tracks into the session ---
        for track_name, track_points in loaded_tracks.items():
            if track_name in self.track_name_to_id:
                track_id = self.track_name_to_id[track_name]
                self.raw_tracks[track_id] = track_points
            else:
                # New track: create a new ID and register it
                track_id = f"T{len(self.raw_tracks) + 1}"
                self.raw_tracks[track_id] = track_points
                self.track_order.append(track_id)
                self.track_name_to_id[track_name] = track_id

        self.refresh_track_listbox()   # Update the All Tracks listbox to show loaded tracks

        # --- Load reversal events if they exist ---
        loaded_reversals = False
        if os.path.isdir(reversal_dir):
            loaded_reversals = self._load_reversal_events_npy(reversal_dir, x_saved, y_saved, z_saved, t_saved)

        if loaded_reversals:
            # Switch to reversal mode and show the reversal events window
            self.mode_var.set("Reversal event")
            self.show_reversal_window()
            self.refresh_reversal_table()
            self.update_image()

        self.dirty = True                    # Mark as modified (loaded data needs eventual save)
        self._ensure_autosave_scheduled()    # Start autosave timer


    # =========================================================================
    # DELETION
    # The Delete key or Delete button removes selected rows from the data table
    # OR removes entire selected tracks from the session.
    # =========================================================================

    def delete_selection(self):
        """
        Deletes the currently selected rows from the data table (if any are selected),
        or deletes the selected track(s) from the All Tracks listbox (if table has no selection).

        Always prompts the user to confirm before deleting.
        """
        # --- Case 1: Rows are selected in the data table ---
        selected_rows = self.track_tree.selection()
        if selected_rows:
            confirm = messagebox.askokcancel(
                "Confirm Deletion",
                f"Are you sure you want to delete the selected {len(selected_rows)} row(s)?"
            )
            if not confirm:
                return

            # Convert row IDs (which are Select# values stored as strings) to integers
            row_ids = [int(iid) for iid in selected_rows]

            # Remove points with matching Select# from the active track's raw data
            self.raw_tracks[self.edit_mode_track] = [
                pt for pt in self.raw_tracks[self.edit_mode_track]
                if pt[0] not in row_ids
            ]

            self.update_track_table()   # Refresh the data table
            self.update_image()         # Redraw image (removes deleted points from overlay)
            self.dirty = True
            self._ensure_autosave_scheduled()
            return

        # --- Case 2: Tracks are selected in the All Tracks listbox ---
        selected_indices = self.track_listbox.curselection()
        if selected_indices:
            # Show confirmation dialog
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

            # Delete each selected track in reverse index order
            # (reverse order prevents index shifting from affecting subsequent deletions)
            for i in reversed(selected_indices):
                track_name = self.track_listbox.get(i)
                track_id = self.track_name_to_id.get(track_name)

                # Remove from all data structures
                if track_id in self.raw_tracks:
                    del self.raw_tracks[track_id]
                if track_id in self.track_order:
                    self.track_order.remove(track_id)
                if track_name in self.track_name_to_id:
                    del self.track_name_to_id[track_name]

                self.track_listbox.delete(i)   # Remove from the listbox

                # If the deleted track was in edit mode, clear the edit mode state
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


    # =========================================================================
    # UI CONSTRUCTION
    # build_ui() creates all the visual elements of the application window:
    # the image canvas, sliders, buttons, data table, and listbox.
    # =========================================================================

    def build_ui(self):
        """
        Constructs and arranges all widgets in the main application window.

        Layout overview:
        ┌─────────────────────────────┬───────────────────────────────┐
        │  LEFT PANEL (left_frame)    │  RIGHT PANEL (track_info_frame)│
        │                             │                                │
        │  [Matplotlib image canvas]  │  Edit Mode Track label         │
        │                             │  Data table (Treeview)         │
        │  [Pixel info label]         │  Unit Conversion panel         │
        │  [Z-plane slider + buttons] │  All Tracks label              │
        │  [Time slider + buttons]    │  Track listbox                 │
        │  [Contrast slider]          │  Mode dropdown                 │
        │  [Progress bar (temp)]      │  Action buttons                │
        │                             │  Opacity slider                │
        └─────────────────────────────┴───────────────────────────────┘
        """
        # ── LEFT PANEL: image display area ──
        # A frame that fills the left portion of the window and expands with it
        self.left_frame = tk.Frame(self.root)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create the matplotlib Figure and Axes for image display
        self.fig = Figure(figsize=(8, 6), dpi=100)   # 8 inches wide, 6 tall, 100 dots-per-inch
        self.ax = self.fig.add_subplot(111)           # A single subplot (1 row, 1 col, position 1)
        self.ax.axis("off")                           # Hide axis lines and tick marks
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)   # Remove padding around the image

        # Embed the matplotlib figure into the tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Connect mouse events on the canvas to handler methods
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_motion)   # Mouse move → show pixel value
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)     # Mouse click → add track point

        # Pixel info label: shows "(x, y) Value: N" as the mouse moves over the image
        self.pixel_info_frame = tk.Frame(self.left_frame, height=25)
        self.pixel_info_frame.pack_propagate(False)   # Don't let children resize the frame
        self.pixel_info_frame.pack(side=tk.TOP, fill=tk.X)
        self.pixel_info_label = tk.Label(self.pixel_info_frame, text="Pixel: N/A", anchor="w")
        self.pixel_info_label.pack(side=tk.LEFT, padx=5, pady=2)

        # Container for Z and time sliders
        self.scrollbars_frame = tk.Frame(self.left_frame)
        self.scrollbars_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # ── Z-PLANE SLIDER ──
        self.z_scroll_container = tk.Frame(self.scrollbars_frame)
        self.z_scroll_container.pack(fill=tk.X, pady=2)

        # Extract all Z values from the subdirectory names to set slider range
        self.z_values = [parse_z_value(d) for d in self.subdirs]
        min_z = min(self.z_values)
        max_z = max(self.z_values)

        # Create the Z-plane slider (a horizontal scale widget)
        self.z_scroll = tk.Scale(
            self.z_scroll_container,
            from_=min_z,        # Minimum Z value (slider left end)
            to=max_z,           # Maximum Z value (slider right end)
            orient=tk.HORIZONTAL,
            label="Z-plane",
            resolution=1,       # Minimum step size = 1 unit
            command=self.on_z_scroll   # Called whenever the slider moves
        )
        self.z_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.z_scroll.set(parse_z_value(self.subdirs[self.sub_index]))   # Set initial position

        # Label showing the current Z value as a number
        self.z_value_label = tk.Label(self.z_scroll_container, text=f"Z: {parse_z_value(self.subdirs[self.sub_index])}")
        self.z_value_label.pack(side=tk.LEFT, padx=5)

        # ◄ and ► navigation buttons for Z-plane
        self.z_prev_button = tk.Button(self.z_scroll_container, text="◄", command=self.z_next)
        self.z_prev_button.pack(side=tk.LEFT)
        self.z_next_button = tk.Button(self.z_scroll_container, text="►", command=self.z_prev)
        self.z_next_button.pack(side=tk.LEFT)

        # ── TIME SLIDER ──
        current_subdir = self.subdirs[self.sub_index]
        max_time = max(len(self.images_by_subdir[current_subdir]) - 1, 0)   # Maximum frame index

        self.t_scroll_container = tk.Frame(self.scrollbars_frame)
        self.t_scroll_container.pack(fill=tk.X, pady=2)

        # Create the Time slider
        self.t_scroll = tk.Scale(
            self.t_scroll_container,
            from_=0,            # First frame
            to=max_time,        # Last frame
            orient=tk.HORIZONTAL,
            label="Time Series (Image Index)",
            command=self.on_t_scroll   # Called whenever the slider moves
        )
        self.t_scroll.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # ◄ and ► navigation buttons for time
        self.t_prev_button = tk.Button(self.t_scroll_container, text="◄", command=self.t_prev)
        self.t_prev_button.pack(side=tk.LEFT)
        self.t_next_button = tk.Button(self.t_scroll_container, text="►", command=self.t_next)
        self.t_next_button.pack(side=tk.LEFT)

        # ── CONTRAST SLIDER ──
        self.contrast_slider = tk.Scale(
            self.left_frame,
            from_=50,       # 50 = no enhancement (left end)
            to=100,         # 100 = full binarization (right end)
            orient=tk.HORIZONTAL,
            label="Contrast",
            command=self.on_contrast_scroll
        )
        self.contrast_slider.set(50)   # Default: no contrast enhancement
        self.contrast_slider.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # ── RIGHT PANEL: track data and controls ──
        # A fixed-width panel on the right side for track info, table, and buttons
        self.track_info_frame = tk.Frame(self.root, bd=2, relief=tk.SUNKEN, width=420)
        self.track_info_frame.pack_propagate(False)   # Prevent contents from resizing the panel
        self.track_info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        # Container for the edit-mode track label and data table
        self.current_track_frame = tk.Frame(self.track_info_frame)
        self.current_track_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── UNIT CONVERSION PANEL ──
        # A small panel with 4 text entry boxes for X, Y, Z, and Time conversion ratios
        self.unit_conv_frame = tk.Frame(self.track_info_frame, bd=1, relief=tk.GROOVE)
        self.unit_conv_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        conv_label = tk.Label(self.unit_conv_frame, text="Unit Conversion", font=("Helvetica", 14, "bold"))
        conv_label.grid(row=0, column=0, columnspan=4, pady=(2, 5))   # Spans all 4 columns

        # Labels for each ratio entry
        tk.Label(self.unit_conv_frame, text="X Ratio:").grid(row=1, column=0, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Y Ratio:").grid(row=1, column=1, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Z Ratio:").grid(row=1, column=2, sticky="e", padx=2)
        tk.Label(self.unit_conv_frame, text="Time Ratio:").grid(row=1, column=3, sticky="e", padx=2)

        # Entry boxes pre-filled with "1.0" (default = no conversion)
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

        # Bind Enter key and focus-loss events to update conversion factors
        # This means the conversion is applied as soon as you press Enter or click elsewhere
        self.x_entry.bind("<Return>", self.update_conversion_factors)
        self.y_entry.bind("<Return>", self.update_conversion_factors)
        self.z_entry.bind("<Return>", self.update_conversion_factors)
        self.t_entry.bind("<Return>", self.update_conversion_factors)

        self.x_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.y_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.z_entry.bind("<FocusOut>", self.update_conversion_factors)
        self.t_entry.bind("<FocusOut>", self.update_conversion_factors)

        # Container for the All Tracks listbox (top portion of made_tracks area)
        self.made_tracks_frame = tk.Frame(self.track_info_frame)
        self.made_tracks_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Container for action buttons (bottom portion)
        self.right_button_frame = tk.Frame(self.track_info_frame)
        self.right_button_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, pady=5)

        # ── EDIT MODE TRACK LABEL ──
        # Shows the name of the track currently in edit mode (e.g., "Edit Mode Track: T2")
        self.track_info_label = tk.Label(
            self.current_track_frame,
            text="Edit Mode Track:",
            anchor="w",
            font=("Helvetica", 16, "bold")
        )
        self.track_info_label.pack(fill=tk.X, padx=5, pady=5)

        # ── DATA TABLE (TREEVIEW) ──
        # A scrollable table that shows all points for the active edit-mode track
        # Columns: Select # | ID # | X | Y | Z | Time
        self.columns = ("select", "id", "x", "y", "z", "t")
        self.track_tree = ttk.Treeview(
            self.current_track_frame,
            columns=self.columns,
            show="headings",       # Only show column headers, not the tree icons on the left
            selectmode="none",     # Disable built-in selection (we handle it manually)
            takefocus=0            # Don't capture keyboard focus (so arrow keys still navigate images)
        )
        self.track_tree.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

        # Set column header labels
        self.track_tree.heading("select", text="Select #")
        self.track_tree.heading("id", text="ID #")
        self.track_tree.heading("x", text="X")
        self.track_tree.heading("y", text="Y")
        self.track_tree.heading("z", text="Z")
        self.track_tree.heading("t", text="Time")

        # Set column widths and center-align all values
        self.track_tree.column("select", width=70, anchor=tk.CENTER)
        self.track_tree.column("id", width=60, anchor=tk.CENTER)
        self.track_tree.column("x", width=70, anchor=tk.CENTER)
        self.track_tree.column("y", width=70, anchor=tk.CENTER)
        self.track_tree.column("z", width=70, anchor=tk.CENTER)
        self.track_tree.column("t", width=80, anchor=tk.CENTER)

        # Style configuration (for potential future theming)
        style = ttk.Style()
        style.configure("selected.Table", background="lightgray")

        # Bind click events to selection handlers
        self.track_tree.bind("<Button-1>", self.on_tree_click_normal)                  # Regular click
        self.track_tree.bind("<Shift-Button-1>", self.on_tree_click_shift, add="+")   # Shift+click (range select)
        self.track_tree.bind("<Control-Button-1>", self.on_tree_click_ctrl, add="+")  # Ctrl+click (toggle select)

        # ── ALL TRACKS LISTBOX ──
        # Shows all created tracks; supports multi-select for overlay display
        track_list_label = tk.Label(self.made_tracks_frame, text="All Tracks:", anchor="w", font=("Helvetica", 16, "bold"))
        track_list_label.pack(fill=tk.X, padx=5, pady=5)

        self.track_listbox = tk.Listbox(self.made_tracks_frame, selectmode=tk.EXTENDED, takefocus=0)
        self.track_listbox.pack(fill=tk.BOTH, padx=5, pady=5, expand=True)

        # Bind interaction events
        self.track_listbox.bind("<Button-1>", self.on_listbox_click)                  # Click (or e+click)
        self.track_listbox.bind("<<ListboxSelect>>", self.on_select_track)             # Selection change
        self.track_listbox.bind("<Control-Button-1>", self.on_listbox_ctrl_click)      # Ctrl+click toggle

        # Prevent Up/Down arrow keys from moving the listbox selection
        # (they should control Z-plane instead)
        self.track_listbox.bind("<Up>", lambda e: self.on_up_key(e))
        self.track_listbox.bind("<Down>", lambda e: self.on_down_key(e))

        # ── MODE DROPDOWN (Tracking vs. Reversal Event) ──
        mode_frame = tk.Frame(self.right_button_frame)
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=(5, 0))
        mode_frame.columnconfigure(1, weight=1)   # Let the dropdown expand

        tk.Label(mode_frame, text="Mode:", font=("Helvetica", 14, "bold")).grid(row=0, column=0, sticky="w")

        # Dropdown menu that switches between "Tracking" and "Reversal event" modes
        mode_menu = tk.OptionMenu(
            mode_frame,
            self.mode_var,           # The StringVar that stores the current mode
            "Tracking",
            "Reversal event",
            command=self.on_mode_change   # Called whenever the selection changes
        )
        mode_menu.config(font=("Helvetica", 12))
        mode_menu.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        # ── ACTION BUTTONS ──
        button_font = ("Helvetica", 16, "bold")

        # Create all the buttons (they will be placed using grid below)
        self.new_track_button = tk.Button(self.right_button_frame, text="Start New Track", font=button_font, command=self.start_new_track)
        self.save_button = tk.Button(self.right_button_frame, text="Save", font=button_font, command=self.save_tracks)
        self.show_tracks_button = tk.Button(self.right_button_frame, text="Show Tracks", font=button_font, command=self.toggle_show_tracks)
        self.home_button = tk.Button(self.right_button_frame, text="🏠 Home", font=button_font, command=self.reset_zoom)
        self.load_tracks_button = tk.Button(self.right_button_frame, text="Load Previous Tracks", font=button_font, command=self.load_previous_tracks)
        self.delete_button = tk.Button(self.right_button_frame, text="Delete", font=button_font, command=self.delete_selection)

        # ── TRACK OPACITY SLIDER ──
        self.translucence_slider = tk.Scale(
            self.right_button_frame,
            from_=0,          # 0% = fully transparent (invisible)
            to=100,           # 100% = fully opaque
            orient=tk.HORIZONTAL,
            label="Track Opacity (%)",
            command=self.on_translucence_change,
            font=("Helvetica", 14)
        )
        self.translucence_slider.set(50)   # Default: 50% opacity

        # Allow buttons to expand equally in both columns
        self.right_button_frame.columnconfigure(0, weight=1)
        self.right_button_frame.columnconfigure(1, weight=1)

        # Place buttons in a 2-column grid layout (row, column)
        self.new_track_button.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.save_button.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        self.show_tracks_button.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        self.home_button.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)
        self.load_tracks_button.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
        self.delete_button.grid(row=3, column=1, sticky="nsew", padx=5, pady=5)
        self.translucence_slider.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        # Bind Ctrl+MouseWheel to zoom (for Windows/Mac where scroll events are different)
        self.canvas.get_tk_widget().bind("<Control-MouseWheel>", self.on_ctrl_mousewheel)


    # =========================================================================
    # REVERSAL EVENT MODE: UI + DATA
    # When in "Reversal event" mode, clicking on a point near a track marks it
    # as a reversal event. These are shown with 'X' markers on the overlay and
    # stored in a separate pop-up window table.
    # =========================================================================

    def on_mode_change(self, selected_value=None):
        """
        Called when the mode dropdown changes (Tracking ↔ Reversal event).

        If switching to Reversal event mode: opens (or shows) the reversal events window.
        If switching back to Tracking mode: does nothing special — reversal data is preserved
        but clicks will add points again instead of marking reversals.

        Refreshes the image overlay so 'X' markers appear/disappear as needed.
        """
        mode = self.mode_var.get()
        if mode == "Reversal event":
            self.show_reversal_window()   # Open the reversal events pop-up

        # Redraw the overlay to update marker styles
        self.update_image()

    def show_reversal_window(self):
        """
        Creates (if it doesn't exist) and shows the "Reversal Events" pop-up window.

        The window contains a table (Treeview) listing all marked reversal events,
        and buttons to delete selected events or hide the window.

        If the window was previously hidden (withdrawn), this brings it back to the front.
        """
        # Create the window only if it doesn't exist yet (or was destroyed)
        if self.reversal_window is None or not tk.Toplevel.winfo_exists(self.reversal_window):
            self.reversal_window = tk.Toplevel(self.root)   # Create a secondary window
            self.reversal_window.title("Reversal Events")
            self.reversal_window.geometry("500x300")         # Initial size

            # Override the close button: hide the window instead of destroying it
            # (so the data is preserved and the window can be reopened)
            self.reversal_window.protocol("WM_DELETE_WINDOW", self.hide_reversal_window)

            # Create the reversal events table with 6 columns
            cols = ("track", "id", "x", "y", "z", "t")
            self.reversal_tree = ttk.Treeview(self.reversal_window, columns=cols, show="headings")
            self.reversal_tree.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            # Set column headers
            self.reversal_tree.heading("track", text="Track")
            self.reversal_tree.heading("id", text="ID #")
            self.reversal_tree.heading("x", text="X")
            self.reversal_tree.heading("y", text="Y")
            self.reversal_tree.heading("z", text="Z")
            self.reversal_tree.heading("t", text="Time")

            # Set column widths
            self.reversal_tree.column("track", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("id", width=55, anchor=tk.CENTER)
            self.reversal_tree.column("x", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("y", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("z", width=70, anchor=tk.CENTER)
            self.reversal_tree.column("t", width=90, anchor=tk.CENTER)

            # Button row at the bottom of the reversal events window
            btn_frame = tk.Frame(self.reversal_window)
            btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

            tk.Button(btn_frame, text="Delete Selected Event", command=self.delete_selected_reversal_event).pack(side=tk.LEFT)
            tk.Button(btn_frame, text="Hide", command=self.hide_reversal_window).pack(side=tk.RIGHT)

        # Show the window (if it was hidden) and bring it to the front
        self.reversal_window.deiconify()
        self.reversal_window.lift()
        self.refresh_reversal_table()   # Populate the table with current data

    def hide_reversal_window(self):
        """
        Hides (but doesn't destroy) the reversal events window.
        The data is preserved; the window can be shown again later.
        """
        if self.reversal_window and tk.Toplevel.winfo_exists(self.reversal_window):
            self.reversal_window.withdraw()   # withdraw() hides without destroying

    def refresh_reversal_table(self):
        """
        Clears and rebuilds the reversal events table from self.reversal_events.
        Values are displayed in converted (real-world) units.
        Events are sorted by (track name, time).
        """
        if not self.reversal_tree:
            return   # Table doesn't exist yet

        # Clear all existing rows from the table
        for iid in self.reversal_tree.get_children():
            self.reversal_tree.delete(iid)

        # Sort events by track name then by time value
        def sort_key(ev):
            return (ev["track"], ev["t"])

        for idx, ev in enumerate(sorted(self.reversal_events, key=sort_key), start=1):
            track = ev["track"]
            select_id = ev["select_id"]

            # Get the temporal ID# for this point (its rank when sorted by time within the track)
            id_num = self.get_temporal_id_for_select_id(track, select_id)

            # Apply unit conversion for display
            conv_x = round(ev["x"] * self.x_ratio, 1)
            conv_y = round(ev["y"] * self.y_ratio, 1)
            conv_z = round(ev["z"] * self.z_ratio, 1)
            conv_t = round(ev["t"] * self.t_ratio, 3)

            # Insert this event as a row (iid = "T2:7" for track T2, select_id 7)
            self.reversal_tree.insert(
                "",
                tk.END,
                iid=f"{track}:{select_id}",
                values=(track, id_num, conv_x, conv_y, conv_z, f"{conv_t:.3f}")
            )

    def get_temporal_id_for_select_id(self, track_id, select_id):
        """
        Returns the temporal ID# of a point — i.e., its rank (1, 2, 3, ...)
        when all points in the track are sorted by increasing time.

        The point is identified by its select_id (the internal unique ID assigned when clicked).
        Returns "" (empty string) if the point can't be found.
        """
        pts = self.raw_tracks.get(track_id, [])
        if not pts:
            return ""

        # Sort all points in the track by their raw time value
        pts_sorted = sorted(pts, key=lambda r: r[4])

        # Find the position of this specific point
        for idx, p in enumerate(pts_sorted, start=1):
            if p[0] == select_id:
                return idx   # Return 1-based position
        return ""

    def delete_selected_reversal_event(self):
        """
        Deletes the selected row(s) from the reversal events table.
        Removes the events from self.reversal_events and self.reversal_point_keys,
        and redraws the image to remove the 'X' markers.
        """
        if not self.reversal_tree:
            return

        sel = self.reversal_tree.selection()   # Get selected row IDs (format: "T2:7")
        if not sel:
            return

        # Confirm deletion
        confirm = messagebox.askokcancel("Confirm Deletion", f"Delete {len(sel)} reversal event(s)?")
        if not confirm:
            return

        # Parse the selected row IDs to extract (track, select_id) pairs
        keys_to_remove = set()
        for iid in sel:
            try:
                track, select_str = iid.split(":")       # Split "T2:7" into "T2" and "7"
                select_id = int(select_str)
                keys_to_remove.add((track, select_id))
            except:
                continue

        # Remove the matching events from the events list
        self.reversal_events = [
            ev for ev in self.reversal_events
            if (ev["track"], ev["select_id"]) not in keys_to_remove
        ]

        # Remove from the fast-lookup set (used for 'X' marker rendering)
        self.reversal_point_keys -= keys_to_remove

        self.refresh_reversal_table()   # Update the table
        self.update_image()             # Redraw image to remove 'X' markers

    def _pick_nearest_point_on_track(self, track_id, click_x, click_y):
        """
        Given a click position (click_x, click_y) in image pixel coordinates,
        finds the closest existing point on the specified track.

        Prefers points that are on the current Z-plane and time frame.
        Falls back to all points if none match the current frame.

        Only returns a point if it's within self.reversal_pick_radius pixels of the click.
        Returns the matched raw point tuple (select_id, x, y, z, t), or None if no match.
        """
        pts = self.raw_tracks.get(track_id, [])
        if not pts:
            return None

        current_z = parse_z_value(self.subdirs[self.sub_index])
        current_t = self.time_index

        # First, try to find candidates that match both current Z and current time
        candidates = [p for p in pts if (p[3] == current_z and p[4] == current_t)]
        if not candidates:
            candidates = pts   # Fall back to all points in the track

        # Find the closest point using Euclidean distance (d² = dx² + dy²)
        best = None
        best_d2 = None
        for p in candidates:
            dx = p[1] - click_x   # Horizontal distance to this point
            dy = p[2] - click_y   # Vertical distance to this point
            d2 = dx*dx + dy*dy    # Squared distance (avoids expensive square root)
            if best is None or d2 < best_d2:
                best = p
                best_d2 = d2

        if best is None:
            return None

        # Reject if the closest point is still too far away (beyond the pick radius)
        if best_d2 is not None and best_d2 > (self.reversal_pick_radius ** 2):
            return None

        return best   # Return the winning point tuple

    def mark_reversal_event_from_click(self, click_x, click_y):
        """
        In "Reversal event" mode: when the user clicks on the image,
        this method finds the nearest point on the selected track and
        marks it as a reversal event.

        Requirements:
          - Exactly one track must be selected in the All Tracks listbox.
          - The click must be within reversal_pick_radius pixels of an existing point.

        If the clicked point is already marked as a reversal event, offers to remove the marking.
        """
        # Enforce single-track selection requirement
        if not self.selected_tracks or len(self.selected_tracks) != 1:
            messagebox.showinfo("Select Track", "Please select exactly one track (e.g., T2) before marking a reversal event.")
            return

        track_id = self.selected_tracks[0]

        # Find the nearest point on that track to the click position
        pt = self._pick_nearest_point_on_track(track_id, click_x, click_y)
        if pt is None:
            messagebox.showinfo("No Point Found", f"No nearby point found on {track_id}. Try clicking closer to an existing point.")
            return

        select_id, x, y, z, t = pt   # Unpack the point tuple
        key = (track_id, select_id)  # Unique identifier for this point

        # Check if this point is already marked as a reversal event
        if key in self.reversal_point_keys:
            # Offer to toggle the marking off
            remove = messagebox.askyesno("Already Marked", "This point is already marked as a reversal event.\nRemove it?")
            if remove:
                self.reversal_point_keys.remove(key)
                self.reversal_events = [ev for ev in self.reversal_events if (ev["track"], ev["select_id"]) != key]
                self.refresh_reversal_table()
                self.update_image()
            return   # Either removed or user said no — done either way

        # Add the new reversal event
        self.reversal_point_keys.add(key)   # Add to fast-lookup set
        self.reversal_events.append({
            "track": track_id,
            "select_id": select_id,
            "x": x,
            "y": y,
            "z": z,
            "t": t
        })

        # Make sure the reversal events window is visible and updated
        self.show_reversal_window()
        self.refresh_reversal_table()
        self.update_image()   # Redraw so 'X' marker appears at this point


    # =========================================================================
    # TRACK OPACITY (TRANSLUCENCE) SLIDER
    # =========================================================================

    def on_translucence_change(self, val_str):
        """
        Called whenever the "Track Opacity (%)" slider is moved.
        Converts the percentage (0–100) to a decimal (0.0–1.0) and redraws the image.

        0% → tracks invisible, 100% → tracks fully opaque.
        """
        try:
            percent = float(val_str)
        except ValueError:
            percent = 100   # Default to fully opaque if parsing fails

        self.tracks_alpha = percent / 100.0   # Convert percentage to 0.0–1.0 range
        self.update_image()                    # Redraw with new opacity


# =============================================================================
# PROGRAM ENTRY POINT
# This block runs only when the script is executed directly (not when imported).
# It creates the main tkinter window, launches the application, and starts
# the tkinter event loop (which keeps the window open and responsive).
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()          # Create the main application window
    root.state("zoomed")    # Start with the window maximized to fill the screen
    app = ImageTrackerApp(root)   # Create the app (this calls __init__ and builds the UI)
    root.mainloop()         # Start the event loop — the program waits here for user input
                            # until the window is closed

