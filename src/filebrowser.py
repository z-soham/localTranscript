"""Custom file browser dialog with list/detailed view and column sorting."""

import os
import tkinter as tk
from pathlib import Path
from tkinter import ttk


class FileBrowserDialog:
    """Popup file browser with list/detailed view toggle and column sorting.

    Parameters
    ----------
    parent : tk.Tk or tk.Toplevel
        Parent window for modality.
    title : str
        Dialog window title.
    filetypes : list[tuple[str, list[str]]] | None
        Filter as [(label, [ext, ...])]. ``None`` shows all files.
    initial_dir : str | Path | None
        Starting directory. ``None`` uses ``os.path.expanduser("~")``.
    """

    def __init__(
        self,
        parent: tk.Tk | tk.Toplevel,
        title: str = "Select File",
        filetypes: list[tuple[str, list[str]]] | None = None,
        initial_dir: str | Path | None = None,
    ):
        self.parent = parent
        self.result: str | None = None
        self._allowed_exts: set[str] | None = None
        self._build_ext_filter(filetypes)

        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("800x500")
        self.dialog.minsize(600, 350)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self._current_dir = Path(initial_dir) if initial_dir else Path.home()

        self._view = tk.StringVar(value="detailed")
        self._sort_col = tk.StringVar(value="modified")
        self._sort_reverse = tk.BooleanVar(value=True)

        self._build_ui()
        self._populate()

        self.dialog.wait_window()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.dialog.columnconfigure(0, weight=1)
        self.dialog.rowconfigure(2, weight=1)

        # -- Path bar --
        path_frame = ttk.Frame(self.dialog, padding=(8, 8, 8, 0))
        path_frame.grid(row=0, column=0, sticky="ew")
        path_frame.columnconfigure(1, weight=1)

        self._path_var = tk.StringVar(value=str(self._current_dir))
        ttk.Button(path_frame, text="Up", command=self._go_up, width=5).grid(row=0, column=0, padx=(0, 4))
        ttk.Entry(path_frame, textvariable=self._path_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(path_frame, text="Go", command=self._navigate_path, width=5).grid(row=0, column=2, padx=(4, 0))

        # -- View toggle --
        view_frame = ttk.Frame(self.dialog, padding=(8, 4, 8, 0))
        view_frame.grid(row=1, column=0, sticky="ew")

        ttk.Label(view_frame, text="View:").grid(row=0, column=0, padx=(0, 6))
        ttk.Radiobutton(view_frame, text="List", variable=self._view, value="list", command=self._apply_view).grid(row=0, column=1, padx=(0, 10))
        ttk.Radiobutton(view_frame, text="Detailed", variable=self._view, value="detailed", command=self._apply_view).grid(row=0, column=2)

        ttk.Label(view_frame, text="Sort:").grid(row=0, column=3, padx=(20, 6))
        self._sort_menu = ttk.Combobox(view_frame, textvariable=self._sort_col, values=["name", "size", "type", "modified"], state="readonly", width=12)
        self._sort_menu.grid(row=0, column=4, padx=(0, 6))
        self._sort_menu.bind("<<ComboboxSelected>>", lambda _: self._populate())

        rev_var = self._sort_reverse

        def _toggle_reverse():
            rev_var.set(not rev_var.get())
            self._populate()

        ttk.Checkbutton(view_frame, text="Desc", variable=rev_var, command=_toggle_reverse).grid(row=0, column=5)

        # -- Treeview --
        tree_frame = ttk.Frame(self.dialog)
        tree_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)

        columns = ("size", "type", "modified")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", selectmode="browse")

        self.tree.heading("name", text="Name", command=lambda c="name": self._on_header_click(c))
        self.tree.heading("size", text="Size", command=lambda c="size": self._on_header_click(c))
        self.tree.heading("type", text="Type", command=lambda c="type": self._on_header_click(c))
        self.tree.heading("modified", text="Modified", command=lambda c="modified": self._on_header_click(c))

        self.tree.column("name", minwidth=200, width=350)
        self.tree.column("size", minwidth=60, width=80, anchor="e")
        self.tree.column("type", minwidth=50, width=70)
        self.tree.column("modified", minwidth=130, width=150)

        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Return>", self._on_double_click)

        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")

        # -- Bottom bar --
        bottom = ttk.Frame(self.dialog, padding=(8, 4, 8, 8))
        bottom.grid(row=3, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self._status_label = ttk.Label(bottom, text="")
        self._status_label.grid(row=0, column=0, sticky="w")

        btn_frame = ttk.Frame(bottom)
        btn_frame.grid(row=0, column=1)
        ttk.Button(btn_frame, text="Select", command=self._on_select, width=10).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btn_frame, text="Cancel", command=self._on_cancel, width=10).grid(row=0, column=1)

    # ------------------------------------------------------------------
    # Directory listing
    # ------------------------------------------------------------------

    def _build_ext_filter(self, filetypes: list[tuple[str, list[str]]] | None) -> None:
        if filetypes is None:
            return
        exts: set[str] = set()
        for _, extensions in filetypes:
            for ext in extensions:
                exts.add(ext.lower().lstrip("*").lower())
        self._allowed_exts = exts

    def _populate(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            entries = list(self._current_dir.iterdir())
        except PermissionError:
            self._status_label.configure(text="Permission denied.")
            return

        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if not e.is_dir()]

        col = self._sort_col.get()
        reverse = self._sort_reverse.get()
        sort_fn = lambda p: self._get_sort_key(p, col)
        dirs.sort(key=sort_fn, reverse=reverse)
        files.sort(key=sort_fn, reverse=reverse)

        for entry in dirs:
            self._add_entry(entry, is_dir=True)
        for entry in files:
            self._add_entry(entry, is_dir=False)

        self._apply_view()
        self._status_label.configure(text=f"{len(dirs)} folders, {len(files)} files")

    def _add_entry(self, path: Path, is_dir: bool) -> None:
        if not is_dir and self._allowed_exts:
            if path.suffix.lower() not in self._allowed_exts:
                return

        name = path.name + ("/" if is_dir else "")
        try:
            stat = path.stat()
            size = stat.st_size if not is_dir else 0
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0

        size_str = _human_size(size) if not is_dir else ""
        ext = path.suffix.lstrip(".").upper() if not is_dir else "Folder"
        from datetime import datetime
        mod_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M") if mtime else ""

        self.tree.insert("", "end", iid=str(path), values=(name, size_str, ext, mod_str))

    def _apply_view(self) -> None:
        mode = self._view.get()
        if mode == "list":
            self.tree.column("size", width=0, minwidth=0, stretch=False)
            self.tree.column("type", width=0, minwidth=0, stretch=False)
            self.tree.column("modified", width=0, minwidth=0, stretch=False)
            self.tree.column("name", width=400)
        else:
            self.tree.column("size", width=80, minwidth=60)
            self.tree.column("type", width=70, minwidth=50)
            self.tree.column("modified", width=150, minwidth=130)
            self.tree.column("name", width=350)

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def _on_header_click(self, col: str) -> None:
        if self._sort_col.get() == col:
            self._sort_reverse.set(not self._sort_reverse.get())
        else:
            self._sort_col.set(col)
            if col == "modified":
                self._sort_reverse.set(True)
            else:
                self._sort_reverse.set(False)
        self._populate()

    def _get_sort_key(self, path: Path, col: str):
        try:
            stat = path.stat()
        except OSError:
            stat = None

        if col == "name":
            return path.name.lower()
        elif col == "size":
            return stat.st_size if stat else 0
        elif col == "type":
            return path.suffix.lower()
        elif col == "modified":
            return stat.st_mtime if stat else 0
        return path.name.lower()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_up(self) -> None:
        self._current_dir = self._current_dir.parent
        self._path_var.set(str(self._current_dir))
        self._populate()

    def _navigate_path(self) -> None:
        target = Path(self._path_var.get().strip())
        if target.is_dir():
            self._current_dir = target.resolve()
            self._path_var.set(str(self._current_dir))
            self._populate()
        else:
            self._status_label.configure(text="Not a valid directory.")

    def _on_double_click(self, event: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        path = Path(selection[0])
        if path.is_dir():
            self._current_dir = path
            self._path_var.set(str(self._current_dir))
            self._populate()
        else:
            self._confirm_selection(path)

    # ------------------------------------------------------------------
    # Selection / dismissal
    # ------------------------------------------------------------------

    def _on_select(self) -> None:
        selection = self.tree.selection()
        if not selection:
            self._status_label.configure(text="No file selected.")
            return
        path = Path(selection[0])
        if path.is_dir():
            self._status_label.configure(text="Cannot select a folder.")
            return
        self._confirm_selection(path)

    def _confirm_selection(self, path: Path) -> None:
        self.result = str(path)
        self.dialog.destroy()

    def _on_cancel(self) -> None:
        self.result = None
        self.dialog.destroy()


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size) < 1024:
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
