import os
import subprocess
import sys


class ToolLauncher:
    """Launch external tool scripts from project tools folder (with fallback)."""

    def __init__(self):
        gui_dir = os.path.dirname(os.path.abspath(__file__))
        # .../Android_GNSS_Analysis/src/gui -> Android_GNSS_Analysis project root
        self.project_root = os.path.abspath(os.path.join(gui_dir, "..", ".."))
        # .../Android_GNSS_Analysis/src/gui -> workspace root (.../Arna)
        self.workspace_root = os.path.abspath(os.path.join(gui_dir, "..", "..", ".."))
        preferred_tools = os.path.join(self.project_root, "tools")
        legacy_tools = os.path.join(self.workspace_root, "tools")
        self.tools_root = preferred_tools if os.path.isdir(preferred_tools) else legacy_tools

    def script_path(self, *parts):
        return os.path.join(self.tools_root, *parts)

    def launch(self, script_path):
        if not os.path.exists(script_path):
            return False, f"脚本不存在: {script_path}"

        try:
            subprocess.Popen(
                [sys.executable, script_path],
                cwd=self.workspace_root,
                shell=False,
            )
            return True, f"已启动: {os.path.basename(script_path)}"
        except Exception as exc:
            return False, f"启动失败: {exc}"
