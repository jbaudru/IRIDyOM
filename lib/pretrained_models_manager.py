"""pretrained_models_manager.py

Manager for organizing and accessing pretrained LTM models.

Directory structure:
    IRIDyOM/pretrained_models/
    ├── datasets/
    │   ├── dataset_name/
    │   │   ├── pitch_augmented_true/
    │   │   │   ├── graphs/
    │   │   │   │   ├── order_1.gpickle
    │   │   │   │   └── ...
    │   │   │   ├── alphabet.json
    │   │   │   ├── target_alphabet.json
    │   │   │   └── metadata.json
    │   │   ├── interval_augmented_false/
    │   │   │   └── ...
    │   │   └── ...
    │   └── ...
    └── graphml_exports/
        ├── dataset_name/
        │   ├── pitch_augmented_true/
        │   │   ├── order_1.graphml
        │   │   └── ...
        │   └── ...
        └── ...
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    """Configuration for a pretrained model location."""
    dataset_name: str
    source_viewpoint: str
    augmented: bool
    target_viewpoint: Optional[str] = None
    
    def __str__(self) -> str:
        """Human-readable representation."""
        aug_str = "with augmentation" if self.augmented else "no augmentation"
        return f"{self.source_viewpoint} ({aug_str})"


class PretrainedModelsManager:
    """Manager for organizing and accessing pretrained LTM models."""
    
    def __init__(self, base_dir: Optional[str] = None):
        """Initialize the manager.
        
        Args:
            base_dir: Base directory for pretrained models. If None, uses:
                     - Development mode: pretrained_models/ (next to this module)
                     - Installed mode: User appdata directory (LOCALAPPDATA on Windows, ~/.iridyom on macOS/Linux)
        """
        if base_dir is None:
            # Default: Try user-writable location first (for installed version)
            if sys.platform == "win32":
                # Windows: Use LOCALAPPDATA/IRIDyOM (fallback to legacy GraphIDYOM)
                base_root = os.getenv("LOCALAPPDATA")
                if base_root:
                    preferred = Path(base_root) / "IRIDyOM" / "pretrained_models"
                    legacy = Path(base_root) / "GraphIDYOM" / "pretrained_models"
                    base_dir = str(legacy if legacy.exists() and not preferred.exists() else preferred)
            else:
                # macOS/Linux: Use ~/.iridyom (fallback to legacy ~/.graphidyom)
                preferred = Path.home() / ".iridyom" / "pretrained_models"
                legacy = Path.home() / ".graphidyom" / "pretrained_models"
                base_dir = str(legacy if legacy.exists() and not preferred.exists() else preferred)
            
            # Fallback to development directory if user-writable dir is not accessible
            if not base_dir:
                this_dir = Path(__file__).resolve().parent
                base_dir = str(this_dir / "pretrained_models")
        
        self.base_dir = Path(base_dir)
        self.datasets_dir = self.base_dir / "datasets"
        self.graphml_dir = self.base_dir / "graphml_exports"
        
        # Ensure directories exist, with graceful fallback if permission denied
        try:
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
            self.graphml_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            # If we can't create directories in the user location, try development location
            this_dir = Path(__file__).resolve().parent
            fallback_base = this_dir / "pretrained_models"
            try:
                fallback_base.mkdir(parents=True, exist_ok=True)
                self.base_dir = fallback_base
                self.datasets_dir = self.base_dir / "datasets"
                self.graphml_dir = self.base_dir / "graphml_exports"
                self.datasets_dir.mkdir(parents=True, exist_ok=True)
                self.graphml_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                # If both fail, just continue without creating directories
                # (application may still work if not saving/loading models)
                pass
    
    def get_model_dir(self, config: ModelConfig) -> Path:
        """Get the directory for a model with given configuration.
        
        Args:
            config: Model configuration
            
        Returns:
            Path to model directory
        """
        aug_str = "augmented_true" if config.augmented else "augmented_false"
        viewpoint_aug = f"{config.source_viewpoint}_{aug_str}"
        
        model_dir = self.datasets_dir / config.dataset_name / viewpoint_aug
        return model_dir
    
    def get_graphml_dir(self, config: ModelConfig) -> Path:
        """Get the directory for GraphML exports of a model.
        
        Args:
            config: Model configuration
            
        Returns:
            Path to GraphML export directory
        """
        aug_str = "augmented_true" if config.augmented else "augmented_false"
        viewpoint_aug = f"{config.source_viewpoint}_{aug_str}"
        
        export_dir = self.graphml_dir / config.dataset_name / viewpoint_aug
        return export_dir
    
    def model_exists(self, config: ModelConfig) -> bool:
        """Check if a model with given configuration exists.
        
        Args:
            config: Model configuration
            
        Returns:
            True if model directory and metadata exist
        """
        model_dir = self.get_model_dir(config)
        metadata_file = model_dir / "metadata.json"
        return metadata_file.exists()
    
    def validate_model_compatible(self, model, dataset_name: str, 
                                  viewpoint_name: str, augmented: bool) -> bool:
        """Validate that a saved model exists and is compatible with current model.
        
        Checks:
        - Model directory and metadata exist
        - Saved orders match current model's orders
        - Saved target viewpoint matches current model's target viewpoint
        
        Args:
            model: GraphIDYOMModel instance to validate against
            dataset_name: Dataset name
            viewpoint_name: Source viewpoint name (e.g., "pitch", "interval")
            augmented: Whether model was trained with augmentation
            
        Returns:
            True if model exists and is compatible, False otherwise
            
        Raises:
            ValueError: If model exists but is incompatible (with detailed message)
        """
        config = ModelConfig(
            dataset_name=dataset_name,
            source_viewpoint=viewpoint_name,
            augmented=augmented,
            target_viewpoint=model.target_viewpoint,
        )
        
        model_dir = self.get_model_dir(config)
        metadata_file = model_dir / "metadata.json"
        
        # Model doesn't exist
        if not metadata_file.exists():
            return False
        
        # Model exists - validate it's compatible
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Check orders compatibility
            saved_orders = tuple(sorted(metadata.get("orders", [])))
            current_orders = tuple(sorted(model.orders))
            if saved_orders != current_orders:
                raise ValueError(
                    f"Order mismatch for {viewpoint_name}: "
                    f"saved={saved_orders}, current={current_orders}. "
                    "Retrain model with matching orders."
                )
            
            # Check target viewpoint compatibility
            saved_target_vp = metadata.get("target_viewpoint")
            if saved_target_vp != model.target_viewpoint:
                raise ValueError(
                    f"Target viewpoint mismatch for {viewpoint_name}: "
                    f"saved={saved_target_vp}, current={model.target_viewpoint}. "
                    "Retrain model with matching target viewpoint."
                )
            
            # Check viewpoint configuration compatibility
            saved_vp_config = metadata.get("viewpoint_config")
            if saved_vp_config:
                from dataclasses import asdict
                current_vp_config = asdict(model.codec.cfg)
                # Compare important viewpoint fields
                important_fields = [
                    'pitch',
                    'octave',
                    'midi_number',
                    'duration',
                    'length',
                    'offset',
                    'interval',
                    'bioi_ratio',
                ]
                mismatches = []
                for field in important_fields:
                    saved_val = saved_vp_config.get(field)
                    current_val = current_vp_config.get(field)
                    if saved_val != current_val:
                        mismatches.append(f"{field}: saved={saved_val}, current={current_val}")
                
                if mismatches:
                    raise ValueError(
                        f"Viewpoint configuration mismatch for {viewpoint_name}: "
                        f"{'; '.join(mismatches)}. "
                        "Retrain model with matching viewpoint configuration."
                    )
            
            return True
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid metadata.json for {viewpoint_name}: {e}")
        except Exception as e:
            raise ValueError(f"Error validating {viewpoint_name}: {e}")
    
    def list_available_models(self, dataset_name: Optional[str] = None) -> dict:
        """List all available pretrained models.
        
        Args:
            dataset_name: If provided, only list models for this dataset.
                         If None, list all datasets and their models.
        
        Returns:
            Dictionary with dataset structure and available models
        """
        result = {}
        
        if not self.datasets_dir.exists():
            return result
        
        if dataset_name:
            # List models for specific dataset
            dataset_dir = self.datasets_dir / dataset_name
            if dataset_dir.exists():
                for viewpoint_dir in dataset_dir.iterdir():
                    if viewpoint_dir.is_dir():
                        metadata_file = viewpoint_dir / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            result[viewpoint_dir.name] = metadata
            return result
        
        # List all datasets and models
        for dataset_dir in self.datasets_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_models = {}
                for viewpoint_dir in dataset_dir.iterdir():
                    if viewpoint_dir.is_dir():
                        metadata_file = viewpoint_dir / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, "r") as f:
                                metadata = json.load(f)
                            dataset_models[viewpoint_dir.name] = metadata
                
                if dataset_models:
                    result[dataset_dir.name] = dataset_models
        
        return result
    
    def print_available_models(self, dataset_name: Optional[str] = None) -> None:
        """Print all available models in a readable format.
        
        Args:
            dataset_name: If provided, only print models for this dataset.
        """
        models = self.list_available_models(dataset_name)
        
        if not models:
            print("No pretrained models found.")
            return
        
        if dataset_name:
            print(f"\nAvailable models for dataset '{dataset_name}':")
            for viewpoint_config, metadata in models.items():
                orders = metadata.get("orders", [])
                target_vp = metadata.get("target_viewpoint", "None")
                print(f"  • {viewpoint_config}")
                print(f"    - Orders: {orders}")
                print(f"    - Target viewpoint: {target_vp}")
        else:
            print("\nAvailable pretrained models:")
            for dataset, viewpoints in models.items():
                print(f"\n  Dataset: {dataset}")
                for viewpoint_config, metadata in viewpoints.items():
                    orders = metadata.get("orders", [])
                    target_vp = metadata.get("target_viewpoint", "None")
                    print(f"    • {viewpoint_config}")
                    print(f"      - Orders: {orders}")
                    print(f"      - Target: {target_vp}")


def get_model_config_from_name(dataset_name: str, viewpoint_folder_name: str) -> ModelConfig:
    """Parse a viewpoint folder name back into a ModelConfig.
    
    Args:
        dataset_name: Name of dataset
        viewpoint_folder_name: Folder name like "pitch_augmented_true"
        
    Returns:
        ModelConfig with parsed settings
    """
    # Parse format: "viewpoint_augmented_true/false"
    parts = viewpoint_folder_name.split("_augmented_")
    if len(parts) != 2:
        raise ValueError(f"Invalid viewpoint folder name: {viewpoint_folder_name}")
    
    source_viewpoint = parts[0]
    augmented = parts[1].lower() == "true"
    
    return ModelConfig(
        dataset_name=dataset_name,
        source_viewpoint=source_viewpoint,
        augmented=augmented,
    )


if __name__ == "__main__":
    # Example usage
    manager = PretrainedModelsManager()
    
    # List all models
    manager.print_available_models()
    
    # Check if a specific model exists
    config = ModelConfig(
        dataset_name="largeWestern_elsass",
        source_viewpoint="pitch",
        augmented=True,
        target_viewpoint="pitchOctave",
    )
    
    if manager.model_exists(config):
        print(f"\n✓ Model exists: {config}")
        print(f"  Directory: {manager.get_model_dir(config)}")
    else:
        print(f"\n✗ Model does not exist: {config}")
