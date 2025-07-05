"""
Factory for creating tracking systems based on configuration.

Provides a centralized way to instantiate different tracking implementations
with fallback capabilities and configuration-based selection.
"""

import logging
from typing import Dict, Type, Optional, List

from .tracking_interface import TemporalTracker
from .deepsort_tracker import DeepSORTTracker, DEEPSORT_AVAILABLE
from .simple_tracker import SimpleBboxTracker
from config import ProcessingConfig

logger = logging.getLogger('wildcams')


class TrackerFactory:
    """
    Factory for creating temporal tracking systems.
    
    Handles instantiation of different tracker types with automatic
    fallback to simpler methods if dependencies are unavailable.
    """
    
    # Registry of available tracker classes
    _tracker_registry: Dict[str, Type[TemporalTracker]] = {
        'deepsort': DeepSORTTracker,
        'simple': SimpleBboxTracker,
        'bbox': SimpleBboxTracker,  # Alias for simple
    }
    
    # Fallback chain for when preferred tracker is unavailable
    _fallback_chain = ['deepsort', 'simple']
    
    @classmethod
    def create_tracker(cls, tracker_type: str, config: ProcessingConfig) -> TemporalTracker:
        """
        Create a tracker instance of the specified type.
        
        Args:
            tracker_type: Type of tracker to create ('deepsort', 'simple', 'auto')
            config: ProcessingConfig with tracking parameters
            
        Returns:
            Initialized TemporalTracker instance
            
        Raises:
            ValueError: If tracker_type is unknown and no fallback available
        """
        # Handle 'auto' selection
        if tracker_type.lower() == 'auto':
            return cls._create_auto_tracker(config)
        
        # Normalize tracker type
        tracker_type = tracker_type.lower()
        
        # Check if requested tracker is available
        if tracker_type in cls._tracker_registry:
            tracker_class = cls._tracker_registry[tracker_type]
            
            try:
                # Special check for DeepSORT availability
                if tracker_type == 'deepsort' and not DEEPSORT_AVAILABLE:
                    logger.warning("🔄 DeepSORT requested but not available, falling back to simple tracker")
                    return cls._create_fallback_tracker(config)
                
                # Create tracker instance
                tracker = tracker_class(config)
                logger.info(f"✅ Created {tracker.tracking_method} tracker")
                return tracker
                
            except Exception as e:
                logger.error(f"❌ Failed to create {tracker_type} tracker: {e}")
                logger.info("🔄 Attempting fallback tracker...")
                return cls._create_fallback_tracker(config)
        
        else:
            logger.warning(f"⚠️ Unknown tracker type '{tracker_type}', attempting fallback")
            return cls._create_fallback_tracker(config)
    
    @classmethod
    def _create_auto_tracker(cls, config: ProcessingConfig) -> TemporalTracker:
        """
        Automatically select best available tracker.
        
        Args:
            config: ProcessingConfig with tracking parameters
            
        Returns:
            Best available TemporalTracker instance
        """
        logger.info("🎯 Auto-selecting tracker based on availability...")
        
        # Try each tracker in fallback chain
        for tracker_type in cls._fallback_chain:
            try:
                if tracker_type == 'deepsort' and not DEEPSORT_AVAILABLE:
                    logger.info("⏭️ Skipping DeepSORT (not available)")
                    continue
                
                tracker_class = cls._tracker_registry[tracker_type]
                tracker = tracker_class(config)
                logger.info(f"✅ Auto-selected {tracker.tracking_method} tracker")
                return tracker
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to create {tracker_type} tracker: {e}")
                continue
        
        # If all else fails, force simple tracker
        logger.warning("🔄 All trackers failed, forcing simple bbox tracker")
        return SimpleBboxTracker(config)
    
    @classmethod
    def _create_fallback_tracker(cls, config: ProcessingConfig) -> TemporalTracker:
        """
        Create fallback tracker (simple bbox tracker).
        
        Args:
            config: ProcessingConfig with tracking parameters
            
        Returns:
            SimpleBboxTracker instance
        """
        logger.info("🔄 Creating fallback simple bbox tracker")
        return SimpleBboxTracker(config)
    
    @classmethod
    def get_available_trackers(cls) -> Dict[str, bool]:
        """
        Get list of available tracker types and their availability.
        
        Returns:
            Dict mapping tracker names to availability status
        """
        availability = {}
        
        for tracker_name in cls._tracker_registry.keys():
            if tracker_name == 'deepsort':
                availability[tracker_name] = DEEPSORT_AVAILABLE
            else:
                # Simple trackers should always be available
                availability[tracker_name] = True
        
        return availability
    
    @classmethod
    def register_tracker(cls, name: str, tracker_class: Type[TemporalTracker]):
        """
        Register a new tracker type.
        
        Args:
            name: Name to register tracker under
            tracker_class: TemporalTracker subclass
        """
        if not issubclass(tracker_class, TemporalTracker):
            raise ValueError(f"Tracker class must inherit from TemporalTracker")
        
        cls._tracker_registry[name.lower()] = tracker_class
        logger.info(f"📝 Registered tracker: {name}")
    
    @classmethod
    def list_trackers(cls) -> List[str]:
        """
        Get list of all registered tracker names.
        
        Returns:
            List of tracker names
        """
        return list(cls._tracker_registry.keys())


# Convenience function for direct usage
def create_tracker(tracker_type: str = 'auto', config: ProcessingConfig = None) -> TemporalTracker:
    """
    Convenience function to create a tracker.
    
    Args:
        tracker_type: Type of tracker ('auto', 'deepsort', 'simple')
        config: ProcessingConfig instance
        
    Returns:
        Initialized TemporalTracker
    """
    if config is None:
        raise ValueError("ProcessingConfig is required")
    
    return TrackerFactory.create_tracker(tracker_type, config)