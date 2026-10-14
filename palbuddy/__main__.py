import os
import sys


def _report_startup_failure(exc):
    """Started with pythonw (double-click), a failing import would otherwise vanish silently."""
    import traceback
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "palbuddy.log")
        with open(log, "a", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    if sys.stderr is not None:
        sys.stderr.write(text)
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Pal Buddy Guy", "Pal Buddy Guy could not start:\n\n%s\n\nIf a package is missing, delete "
                             "the .venv folder and run PalBuddyGuy.bat again." % exc)
        root.destroy()
    except Exception:
        pass


try:
    from .cli import main
except Exception as e:  # e.g. numpy missing from a broken environment
    _report_startup_failure(e)
    sys.exit(1)

try:
    sys.exit(main())
except SystemExit:
    raise
except Exception as e:
    _report_startup_failure(e)
    sys.exit(1)
