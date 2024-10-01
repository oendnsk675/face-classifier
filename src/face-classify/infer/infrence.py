from .base import BaseInference

class Infrence(BaseInference):
    def __init__(self, device):
        path_to_save = str(self._get_cache_dir()) + "/model.pth"
        self.model_path = Path(path_to_save)
        if not self.model_path.exists():
                raise FileNotFoundError("Model does not exist, check your model location and read README for more information")
        self.device = device
        self.class_dict = []

    def _get_cache_dir(self):
        if sys.platform.startswith("win"):
            cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OnepieceClassifyCache"
        else:
            cache_dir = Path.home() / ".cache" / "OnepieceClassifyCache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir