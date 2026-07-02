"""DeepFaune inference engine for animal vs non-animal classification."""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from config.processing_config import ProcessingConfig
from core.data_types import InferenceResult

logger = logging.getLogger("wildcams")


class DeepFauneInference:
    """
    DeepFaune inference engine for animal vs non-animal classification.

    DeepFaune is specialized for distinguishing wildlife from non-wildlife objects
    in camera trap images, making it ideal for filtering false positives.
    """

    def __init__(self, config: ProcessingConfig, cache_dir: Path | None = None):
        """
        Initialize DeepFaune inference engine.

        Args:
            config: ProcessingConfig with classification parameters
            cache_dir: Optional directory for model caching
        """
        self.config = config
        self.cache_dir = cache_dir or Path("./models_cache")
        self.model: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """Load DeepFaune model."""
        try:
            import torch
            import torchvision.transforms as transforms

            logger.info("🦌 Loading DeepFaune model...")

            # Load DeepFaune model following the actual demo code approach
            import timm

            logger.info("🦌 Loading DeepFaune model from HuggingFace...")

            # Download DeepFaune model from HuggingFace (following the actual demo code)
            from huggingface_hub import hf_hub_download

            model_filename = "deepfaune-vit_large_patch14_dinov2.lvd142m.v3.pt"
            model_path = hf_hub_download(
                repo_id="Addax-Data-Science/Deepfaune_v1.3",
                filename=model_filename,
                cache_dir=str(self.cache_dir),
                local_files_only=False,
            )

            # Create the model using the exact same structure as demo code
            import torch.nn as nn

            # Define the Model class exactly as in demo code
            class Model(nn.Module):
                def __init__(self, device=None):
                    super().__init__()
                    BACKBONE = "vit_large_patch14_dinov2.lvd142m"
                    txt_animalclasses_fr = [
                        "bison",
                        "blaireau",
                        "bouquetin",
                        "castor",
                        "cerf",
                        "chamois",
                        "chat",
                        "chevre",
                        "chevreuil",
                        "chien",
                        "daim",
                        "ecureuil",
                        "elan",
                        "equide",
                        "genette",
                        "glouton",
                        "herisson",
                        "lagomorphe",
                        "loup",
                        "loutre",
                        "lynx",
                        "marmotte",
                        "micromammifere",
                        "mouflon",
                        "mouton",
                        "mustelide",
                        "oiseau",
                        "ours",
                        "ragondin",
                        "raton laveur",
                        "renard",
                        "renne",
                        "sanglier",
                        "vache",
                    ]

                    self.base_model = timm.create_model(
                        BACKBONE, pretrained=False, num_classes=len(txt_animalclasses_fr), dynamic_img_size=True
                    )
                    self.backbone = BACKBONE
                    self.nbclasses = len(txt_animalclasses_fr)
                    self.device = device

                def forward(self, input):
                    x = self.base_model(input)
                    return x

                def loadWeights(self, path):
                    params = torch.load(path, map_location=self.device, weights_only=False)
                    args = params["args"]
                    if self.nbclasses != args["num_classes"]:
                        raise Exception(
                            "You load a model ({}) that does not have the same number of class({})".format(
                                args["num_classes"], self.nbclasses
                            )
                        )
                    self.backbone = args["backbone"]
                    self.nbclasses = args["num_classes"]
                    self.load_state_dict(params["state_dict"])

            # Create and load the model exactly as in demo code
            self.model = Model(device="cpu")
            self.model.loadWeights(model_path)
            self.model.eval()

            logger.info(f"✅ DeepFaune model loaded: {self.model.backbone}, {self.model.nbclasses} classes")

            # DeepFaune uses DINOv2 ViT preprocessing (from demo code)
            from torch import tensor
            from torchvision.transforms import InterpolationMode

            CROP_SIZE = 182  # From demo code
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=(CROP_SIZE, CROP_SIZE),
                        interpolation=InterpolationMode.BICUBIC,
                        max_size=None,
                        antialias=None,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250])),
                ]
            )

            logger.info("✅ DeepFaune model loaded successfully from Hugging Face")

        except ImportError as e:
            logger.error(f"❌ DeepFaune dependencies not available: {e}")
            raise RuntimeError(
                f"❌ DeepFaune configured but dependencies missing: {e}. Install with: pip install torch torchvision requests"
            ) from e

        except Exception as e:
            logger.error(f"❌ Failed to load DeepFaune model: {e}")
            raise RuntimeError(f"❌ DeepFaune model configured but failed to load: {e}") from e

    def classify(self, image_crop: np.ndarray) -> InferenceResult:
        """
        Classify an image crop using DeepFaune - returns generic interface.

        Args:
            image_crop: RGB image array of shape (H, W, 3)

        Returns:
            InferenceResult with generic interface
        """
        return self._classify(image_crop)

    def _classify(self, image_crop: np.ndarray) -> InferenceResult:
        """
        Classify an image crop using DeepFaune.

        Args:
            image_crop: RGB image array of shape (H, W, 3)

        Returns:
            DeepFauneResult with animal vs non-animal classification
        """
        start_time = time.time()

        if not self.model:
            raise RuntimeError("DeepFaune model is not loaded - initialization failed")

        try:
            # Direct model loading from Hugging Face
            import torch
            from PIL import Image

            # Convert numpy array to PIL Image
            if len(image_crop.shape) == 3 and image_crop.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                image_rgb = image_crop[:, :, ::-1]
            else:
                image_rgb = image_crop

            pil_image = Image.fromarray(image_rgb.astype("uint8"))

            # Apply preprocessing transformations
            input_tensor = self.transform(pil_image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                # DeepFaune model outputs logits for 34 animal species classes
                logits = self.model(input_tensor)

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=1)

                # Get all probabilities
                all_probs = probs[0].tolist()

                # Use the class names from the demo code (English)
                species_names = [
                    "bison",
                    "badger",
                    "ibex",
                    "beaver",
                    "red deer",
                    "chamois",
                    "cat",
                    "goat",
                    "roe deer",
                    "dog",
                    "fallow deer",
                    "squirrel",
                    "moose",
                    "equid",
                    "genet",
                    "wolverine",
                    "hedgehog",
                    "lagomorph",
                    "wolf",
                    "otter",
                    "lynx",
                    "marmot",
                    "micromammal",
                    "mouflon",
                    "sheep",
                    "mustelid",
                    "bird",
                    "bear",
                    "nutria",
                    "raccoon",
                    "fox",
                    "reindeer",
                    "wild boar",
                    "cow",
                ]

                # Get top 5 species predictions
                top_indices = torch.topk(probs, k=5, dim=1)[1][0].tolist()
                top_probs = torch.topk(probs, k=5, dim=1)[0][0].tolist()

                # Get the maximum probability (best animal class)
                max_prob = max(all_probs)
                best_species_idx = all_probs.index(max_prob)

                # Log top species predictions with class indices and probabilities
                top_species_info = []
                for j, i in enumerate(top_indices):
                    species = species_names[i] if i < len(species_names) else f"class_{i}"
                    prob = top_probs[j]
                    top_species_info.append(f"{species}({i}):{prob:.3f}")

                logger.info(f"🦌 DeepFaune top5: {', '.join(top_species_info)}")
                logger.info(
                    f"🦌 DeepFaune raw probs shape: {probs.shape}, max_idx: {best_species_idx}, max_prob: {max_prob:.3f}"
                )

                # DeepFaune threshold logic: if the max probability is above threshold, it's an animal
                is_animal = max_prob > self.config.deepfaune_threshold
                animal_confidence = max_prob
                1.0 - max_prob

            processing_time = time.time() - start_time

            logger.info(f"🦌 DeepFaune: animal={is_animal} (conf={animal_confidence:.3f})")

            return InferenceResult(
                model_name="DeepFaune",
                is_animal=is_animal,
                animal_confidence=animal_confidence,
                species=None,  # DeepFaune doesn't do species identification
                species_confidence=0.0,
                can_identify_species=False,
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"❌ DeepFaune inference failed: {e}")
            raise RuntimeError(f"DeepFaune inference failed: {e}") from e

    def get_model_info(self) -> dict:
        """Get information about the DeepFaune model."""
        return {
            "name": "DeepFaune",
            "purpose": "Animal vs Non-Animal Classification",
            "threshold": self.config.deepfaune_threshold,
            "description": "Specialized wildlife detection for camera trap images",
        }
