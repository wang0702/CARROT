from typing import Dict, Any, List, Optional
from .logger import get_carrot_logger

logger = get_carrot_logger(__name__)


class ConfigValidationError(Exception):
    """Custom exception for configuration validation errors."""
    pass


class MCTSConfigValidator:
    """Validator for MCTS configuration parameters."""
    
    # Define valid ranges and types for MCTS parametersb
    PARAMETER_SPECS = {
        'C': {
            'type': (int, float),
            'min_value': 0.1,
            'max_value': 10.0,
            'description': 'UCB1 exploration parameter'
        },
        'lambda_': {
            'type': (int, float),
            'min_value': 0.0,
            'max_value': 1.0,
            'description': 'Cost-constraint parameter'
        },
        'max_iterations': {
            'type': int,
            'min_value': 1,
            'max_value': 1000,
            'description': 'Maximum MCTS iterations'
        },
        'budget': {
            'type': int,
            'min_value': 64,
            'max_value': 8192,
            'description': 'Token budget constraint'
        }
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], 
                       strict: bool = True) -> Dict[str, Any]:
        """
        Validate MCTS configuration parameters.
        
        Args:
            config: Configuration dictionary to validate
            strict: If True, raise exception on invalid parameters
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigValidationError: If validation fails and strict=True
        """
        validated_config = config.copy()
        errors = []
        warnings = []
        
        # Check required parameters
        required_params = ['C', 'lambda_', 'max_iterations']
        for param in required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Validate each parameter
        for param_name, value in config.items():
            if param_name not in cls.PARAMETER_SPECS:
                warnings.append(f"Unknown parameter: {param_name}")
                continue
            
            spec = cls.PARAMETER_SPECS[param_name]
            
            # Type validation
            if not isinstance(value, spec['type']):
                expected_types = [t.__name__ for t in (spec['type'] if isinstance(spec['type'], tuple) else [spec['type']])]
                errors.append(f"Parameter {param_name} must be of type {' or '.join(expected_types)}, got {type(value).__name__}")
                continue
            
            # Range validation
            if 'min_value' in spec and value < spec['min_value']:
                errors.append(f"Parameter {param_name} must be >= {spec['min_value']}, got {value}")
            
            if 'max_value' in spec and value > spec['max_value']:
                errors.append(f"Parameter {param_name} must be <= {spec['max_value']}, got {value}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        # Handle errors
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            if strict:
                raise ConfigValidationError(error_msg)
            else:
                logger.error(error_msg)
                return None
        
        # Set default values for optional parameters
        if 'budget' not in validated_config:
            validated_config['budget'] = 1024
            logger.info("Setting default budget to 1024 tokens")
        
        logger.info("Configuration validation passed")
        return validated_config
    
    @classmethod
    def suggest_optimal_ranges(cls) -> Dict[str, Dict[str, Any]]:
        """Suggest optimal parameter ranges based on research."""
        return {
            'C': {
                'recommended_range': (1.5, 3.0),
                'default': 2.0,
                'description': 'Controls exploration vs exploitation balance'
            },
            'lambda_': {
                'recommended_range': (0.0, 0.5),
                'default': 0.2,
                'description': 'Controls cost constraint importance'
            },
            'max_iterations': {
                'recommended_range': (20, 50),
                'default': 30,
                'description': 'More iterations = better solutions but slower'
            },
            'budget': {
                'recommended_range': (512, 2048),
                'default': 1024,
                'description': 'Token budget for retrieved context'
            }
        }
    
    @classmethod
    def generate_sample_configs(cls, count: int = 5) -> List[Dict[str, Any]]:
        """Generate sample valid configurations for testing."""
        import random
        
        configs = []
        suggestions = cls.suggest_optimal_ranges()
        
        for _ in range(count):
            config = {}
            for param, spec in suggestions.items():
                min_val, max_val = spec['recommended_range']
                if param == 'max_iterations':
                    config[param] = random.randint(int(min_val), int(max_val))
                else:
                    config[param] = round(random.uniform(min_val, max_val), 2)
            configs.append(config)
        
        return configs


def validate_mcts_config(config: Dict[str, Any], 
                        strict: bool = True) -> Optional[Dict[str, Any]]:
    """Convenience function to validate MCTS configuration."""
    return MCTSConfigValidator.validate_config(config, strict)


def get_default_mcts_config() -> Dict[str, Any]:
    """Get a default MCTS configuration."""
    suggestions = MCTSConfigValidator.suggest_optimal_ranges()
    return {param: spec['default'] for param, spec in suggestions.items()}