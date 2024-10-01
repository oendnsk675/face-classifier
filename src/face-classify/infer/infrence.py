from .base import BaseInference
import torch
import numpy as np
from PIL import  Image
from typing import Dict, Optional, Tuple

class Infrence(BaseInference):
    def __init__(self, device):
        path_to_save = str(self._get_cache_dir()) + "/model.pth"
        self.model_path = Path(path_to_save)
        if not self.model_path.exists():
                raise FileNotFoundError("Model does not exist, check your model location and read README for more information")
        self.device = device
        self.class_dict = ["Ahegao", "Angry", "Happy", "Neutral", "Sad", "Surprise"]
        self.nclass = len(class_dict)
        self.model = self._build_model()


    def _get_cache_dir(self):
        if sys.platform.startswith("win"):
            cache_dir = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OnepieceClassifyCache"
        else:
            cache_dir = Path.home() / ".cache" / "OnepieceClassifyCache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _build_model(self):
        state_dict = torch.load(self.model_path, map_location=self.device)
        model_backbone = image_recog(self.nclass)
        model_backbone.load_state_dict(state_dict)
        return model_backbone

    def pre_process(self, image: Optional[str | np.ndarray | Image.Image]) -> torch.Tensor:
        trans = get_test_transforms()

        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img = trans(img).unsqueeze(0)

        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            img = trans(img).unsqueeze(0)

        elif isinstance(image, np.ndarray):
            img = image.astype(np.uint8)
            img = Image.fromarray(img).convert("RGB")
            img = trans(img).unsqueeze(0)

        else:
            print("Image type not recognized")

        return img.to(self.device)

    def forward(self, image_tensor: torch.Tensor) -> torch.Tensor :
        self.model.eval()

        result = self.model(image_tensor)
        return result

    def post_process(self, output: torch.Tensor) -> Tuple[str, float]:
        logits_prob = torch.softmax(output, dim=1)
        class_idx = int(torch.argmax(logits_prob))

        class_name = self.class_dict[class_idx]
        confidence = logits_prob[class_idx]

        return (class_name, float(confidence))

    def forward(self, image: Optional[str | np.ndarray | Image.Image]) -> Dict[str, str]:
        tensor_img = self.pre_process(image=image)
        logits = self.forward(tensor_img)
        class_name, confidance = self.post_process(logits)

        return {"class_name": class_name, "confidance": f"{confidance:.4f}"}

