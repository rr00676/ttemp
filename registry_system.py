from typing import Dict, Any, List, Optional, TypeVar, Generic, Protocol, runtime_checkable, Callable, Type
from dataclasses import dataclass, field
from typing import get_type_hints, cast
import importlib

T = TypeVar('T')  # Generic type for registry items

@dataclass
class RegistryItem(Generic[T]):
    """Container for a registered item with metadata."""
    item: T
    name: str
    description: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@runtime_checkable
class RegistryProtocol(Protocol[T]):
    """Protocol defining registry functionality."""
    
    name: str
    
    def register(self, name: str, item: T, description: str = "", 
                tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
                overwrite: bool = False) -> None:
        """Register an item with optional metadata."""
        ...
    
    def get(self, name: str) -> RegistryItem[T]:
        """Retrieve an item with metadata by name."""
        ...
    
    def list(self) -> List[str]:
        """List all registered item names."""
        ...
    
    def list_with_tags(self, tags: List[str], match_all: bool = False) -> List[str]:
        """List items matching the given tags."""
        ...
    
    def exists(self, name: str) -> bool:
        """Check if an item with the given name exists."""
        ...
    
    def remove(self, name: str) -> None:
        """Remove an item from the registry by name."""
        ...
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for an existing item."""
        ...
    
    def clear(self) -> None:
        """Remove all items from the registry."""
        ...

@dataclass
class Registry(Generic[T]):
    """Registry implementation using dataclasses."""
    name: str
    _items: Dict[str, RegistryItem[T]] = field(default_factory=dict)
    _type_check: Optional[Callable[[Any], bool]] = None
    
    def register(self, name: str, item: T, description: str = "", 
                tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
                overwrite: bool = False) -> None:
        """Register an item with optional metadata."""
        # Type validation if type_check is provided
        if self._type_check is not None and not self._type_check(item):
            raise TypeError(f"Item does not match required type for registry '{self.name}'")
        
        # Check if item already exists
        if name in self._items and not overwrite:
            raise ValueError(f"Item with name '{name}' already exists in registry '{self.name}'")
        
        # Create registry item
        registry_item = RegistryItem(
            item=item,
            name=name,
            description=description,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Add to registry
        self._items[name] = registry_item
    
    def get(self, name: str) -> RegistryItem[T]:
        """Retrieve an item with metadata by name."""
        if name not in self._items:
            raise KeyError(f"No item with name '{name}' in registry '{self.name}'")
        
        return self._items[name]
    
    def list(self) -> List[str]:
        """List all registered item names."""
        return list(self._items.keys())
    
    def list_with_tags(self, tags: List[str], match_all: bool = False) -> List[str]:
        """List items matching the given tags."""
        if match_all:
            return [
                name for name, item in self._items.items()
                if all(tag in item.tags for tag in tags)
            ]
        else:
            return [
                name for name, item in self._items.items()
                if any(tag in item.tags for tag in tags)
            ]
    
    def exists(self, name: str) -> bool:
        """Check if an item with the given name exists."""
        return name in self._items
    
    def remove(self, name: str) -> None:
        """Remove an item from the registry by name."""
        if name not in self._items:
            raise KeyError(f"No item with name '{name}' in registry '{self.name}'")
        
        del self._items[name]
    
    def update_metadata(self, name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for an existing item."""
        if name not in self._items:
            raise KeyError(f"No item with name '{name}' in registry '{self.name}'")
        
        self._items[name].metadata.update(metadata)
    
    def clear(self) -> None:
        """Remove all items from the registry."""
        self._items.clear()
    
    def __len__(self) -> int:
        """Get the number of items in the registry."""
        return len(self._items)

# Factory functions for creating registries
def create_registry(name: str, item_type: type = Any) -> Registry:
    """Create a registry with optional type checking."""
    return Registry(
        name=name,
        _type_check=lambda x: isinstance(x, item_type) if item_type is not Any else True
    )

# Helper functions for registry operations
def register_from_module(registry: Registry, module: str, prefix: str = "", 
                         tags: Optional[List[str]] = None) -> None:
    """Register all matching functions/classes from a module."""
    import inspect
    import warnings
    
    # Import the module dynamically
    module_obj = importlib.import_module(module)
    
    for name, item in inspect.getmembers(module_obj):
        if name.startswith('_'):  # Skip private members
            continue
            
        if name.startswith(prefix) and (inspect.isfunction(item) or inspect.isclass(item)):
            try:
                docstring = inspect.getdoc(item) or ""
                registry.register(
                    name, 
                    item, 
                    description=docstring,
                    tags=tags
                )
            except (ValueError, TypeError) as e:
                warnings.warn(f"Could not register {name} from {module}: {e}")