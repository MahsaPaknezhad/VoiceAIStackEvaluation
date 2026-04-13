import json
import os
from typing import Dict


class ConfigurationManager:
    """
    Manages loading and processing of service configuration files.

    Responsibilities:
    - Load JSON configuration files
    - Substitute environment variables in config values
    - Validate configuration structure
    """

    def __init__(self):
        """Initialize the configuration manager."""
        pass

    def load_config(self, config_path: str) -> Dict:
        """
        Load service configuration from JSON file.

        Args:
            config_path: Path to the JSON configuration file

        Returns:
            Dict containing the loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file contains invalid JSON
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                "Configuration file not found: {config_path}"
            )
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in config file {config_path}: {e}"
            )

    def substitute_env_vars(self, config: Dict) -> Dict:
        """
        Substitute environment variables in configuration values.

        Supports formats:
        - ${VAR_NAME} - Simple variable substitution
        - ${VAR_NAME}:suffix - Variable with suffix (e.g., for ports)

        Args:
            config: Configuration dictionary to process

        Returns:
            Dict with environment variables substituted
        """
        result = config.copy()
        for key, value in result.items():
            if isinstance(value, str) and \
                    value.startswith('${') and value.endswith('}'):
                env_var = value[2:-1]  # Remove ${ and }
                if ':' in env_var:  # Handle ${VAR}:port format
                    env_var, suffix = env_var.split(':', 1)
                    env_value = os.getenv(env_var)
                    if env_value:
                        result[key] = f"{env_value}:{suffix}"
                else:
                    env_value = os.getenv(env_var)
                    if env_value:
                        result[key] = env_value
        return result
