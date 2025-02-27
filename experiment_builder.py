from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
from registry_system import Registry
from utils import call_with_filtered

@dataclass
class ExperimentBuilder:
    """Helper for building experiments with field objects."""
    config: Any  # ExperimentConfig would go here
    
    # Reference to registries
    properties_registry: Registry  # For object properties
    motion_registry: Registry      # For motion functions
    
    # Collection of field objects
    field_objects: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize field objects collection."""
        self.field_objects = []
        self.field_motion = []
        self.object_counts = {}

    def add_objects(self, count: int, object_class: str, object_type: str, 
                   motion_type: str, motion_params: Dict[str, Any] = None) -> List[str]:
        """Add multiple field objects of the same type with the same motion."""
        object_ids = []
        
        # Track only by class (not type)
        if object_class not in self.object_counts:
            self.object_counts[object_class] = 0
            
        current_count = self.object_counts[object_class]
        
        for i in range(count):
            # Use the running count for ID generation
            idx = current_count + i + 1
            object_id = f"{object_type}_{object_class}_{idx}"
            
            self.field_objects.append({
                "id": object_id,
                "object_class": object_class,
                "type": object_type,
                "motion_type": motion_type,
                "motion_params": motion_params if motion_params else {},
            })
            
            object_ids.append(object_id)
            
        # Update the count after adding objects
        self.object_counts[object_class] += count
        
        motion_params = {**motion_params, **self.config.__dict__}
        self.field_motion.append([self.motion_registry._items[motion_type].item, motion_params])
        return object_ids
    
    def build(self) -> Dict[str, Any]:
        """Build the experiment configuration."""
        return {'field_objects': self.field_objects, 
                'field_motion': self.field_motion,
                'object_counts': self.object_counts}
