def center_window(window, width=800, height=600):
    """Center a Tk window on screen. Window may be a Tk or Toplevel."""
    try:
        window.update_idletasks()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = int((screen_width - width) / 2)
        y = int((screen_height - height) / 2)
        window.geometry(f"{width}x{height}+{x}+{y}")
    except Exception:
        # In headless environments these calls may fail; ignore silently
        pass
