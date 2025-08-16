"""Example weather agent using OpenAI and HTTP requests to fetch weather data."""

import argparse
import logging
import tomllib
import sys
from dataclasses import dataclass
from enum import Enum
from os import environ
from typing import Any, Optional, Dict
from pathlib import Path

from strands.models import Model
from dotenv import load_dotenv

load_dotenv()

# Environment variable for application config directory
ENV_CONFIG_DIR="APP_CONFIG_DIR"

class ApplicationConfig:
    """Centralized configuration for the application.

    It essentially is a wrapper around the config.toml file and environment variables.
    """
    def __init__(self, config_dir: str):
        self.config_dir = config_dir
        self.config_path = get_env(
            "APP_CONFIG_DIR",
            Path(__file__).parent / "config",
            cast=Path
        ) / "config.toml"
        self._config = None

    @property
    def config(self) -> Dict[str, Any]:
        if not self._config:
            self._config = self._load_config(self.config_path)
        return self._config

    @property
    def logging(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get("logging", {})

    @property
    def agents(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config["agents"]

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key."""
        return self.config.get(key, default)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from config/config.toml."""
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    def load_agent_config(self, agent_key: str, agent_type: str, fallback_port: int = 8888) -> Dict[str, Any]:
        """
        Load and validate configuration for a specific agent.

        Args:
        agent_key: The agent key (e.g., "calculator", "time", "time-and-calculator")
        agent_type: The agent type ("configurable_agent" or "coordinator_agent")
        fallback_port: The default port to use if not specified in config (default: 8888)
    
        Returns:
            The agent configuration dictionary with port fallback applied
            
        Raises:
            ValueError: If the agent key is not found in configuration
            SystemExit: If configuration validation fails
        """
        agents_config = self.config.get(agent_type, {})

        if agent_key not in agents_config:
            available_keys = list(agents_config.keys())
            logger = logging.getLogger(__name__)
            logger.error(f"Agent '{agent_key}' not found in {agent_type} configuration.")
            logger.error(f"Available {agent_type} agents: {available_keys}")
            sys.exit(1)
        
        agent_config = agents_config[agent_key].copy()  # Copy to avoid modifying original config
        
        # Apply fallback port if not specified
        if "port" not in agent_config:
            agent_config["port"] = fallback_port
            logger = logging.getLogger(__name__)
            logger.warning(f"No port specified for {agent_type}.{agent_key}, using fallback port {fallback_port}")
        
        return agent_config

def get_env(key: str, default: Optional[Any] = None, required: bool = False, cast: Any = str) -> Any:
    """Get an environment variable with optional casting and default value.
    Args:
        key: The environment variable key.
        default: The default value if the environment variable is not set.
        required: Whether the environment variable is required.
        cast: The type to cast the environment variable value to.

    Returns:
        The value of the environment variable, cast to the specified type.

    Raises:
        ValueError: If the environment variable is required but not set, or if the value cannot be cast to the specified type.
    """
    val = environ.get(key, default)
    if required and val is None:
        raise ValueError(f"Missing required environment variable: {key}")
    if val is not None and cast is not None:
        try:
            return cast(val)
        except Exception as ex:
            raise ValueError(f"Invalid value for {key}: {val}") from ex
    return val


_application_config: ApplicationConfig = None

def get_application_config() -> ApplicationConfig:
    """Lazy get or create the application configuration.
    
    Returns:
        The application configuration
    """
    global _application_config
    if _application_config is None:
        _application_config = ApplicationConfig(environ.get(ENV_CONFIG_DIR))
    return _application_config


def parse_agent_key(agent_type: str, description: str = None) -> str:
    """
    Parse command line arguments to get the agent key.
    
    Args:
        agent_type: The type of agent (e.g., "configurable", "coordinator")
        description: Optional description for the argument parser
    
    Returns:
        The agent key from command line arguments
    """
    if description is None:
        description = f"Start a {agent_type} A2A agent"
    
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
python {agent_type}_agent.py calculator         # Start the calculator agent
python {agent_type}_agent.py time              # Start the time agent
        """
    )
    
    parser.add_argument(
        "agent_key",
        help=f"The {agent_type} agent key from config/config.toml"
    )
    
    args = parser.parse_args()
    return args.agent_key


def configure_logging():
    """Configure logging from config/config.toml."""
    try:
        log_level = get_application_config().logging.get("level", "INFO").upper()
        logging.basicConfig(level=getattr(logging, log_level), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
        logging.getLogger().setLevel(getattr(logging, log_level))
    except Exception:
        logging.basicConfig(level=logging.INFO)

# Use centralized logging configuration
configure_logging()
logger = logging.getLogger(__name__)


DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 1000
DEFAULT_AI_MODEL: str = "gpt-4.1"


class ModelType(Enum):
    """Supported model types."""
    OPENAI = "openai"
    LITELLM = "litellm"
    BEDROCK = "bedrock"


@dataclass
class ModelConfig:
    """Configuration for the AI model."""
    model_type: ModelType
    api_key: str
    base_url: Optional[str]
    model_id: str
    max_tokens: int
    temperature: float

    def __post_init__(self):
        """Validate the model configuration after initialization."""
        if not self.model_type:
            raise ValueError("Model type is required (openai|litellm).")
        if not self.api_key:
            raise ValueError("API key is required.")
        if not self.model_id:
            raise ValueError("Model ID is required.")
        if self.max_tokens <= 0:
            raise ValueError("Max tokens must be positive.")
        if not (0 <= self.temperature <= 1):
            raise ValueError("Temperature must be between 0 and 1.")

    @classmethod
    def from_config(cls, model_config: str | None) -> "ModelConfig":
        """Create ModelConfig from environment variables.
        If not model_config is provided, the default one configured in the application config
        will be loaded.

        Args:
            model_config: The model configuration string from the environment.

        Returns: The created ModelConfig instance.
        """
        if not model_config:
            model_config = get_application_config().agents["model"]

        model_type_str, model_id = model_config.split("|", 1)
        try:
            model_type = ModelType(model_type_str)
        except ValueError as ex:
            raise ValueError(f"Unsupported model type: {model_type_str}") from ex

        max_tokens = get_env("MODEL_MAX_TOKEN", DEFAULT_MAX_TOKENS, cast=int)
        temperature = get_env("MODEL_TEMPERATURE", DEFAULT_TEMPERATURE, cast=float)

        match model_type:
            case ModelType.LITELLM:
                return cls(
                    model_type=model_type,
                    api_key=get_env("LITELLM_API_KEY", required=True),
                    base_url=get_env("LITELLM_BASE_URL"),
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case ModelType.OPENAI:
                return cls(
                    model_type=model_type,
                    api_key=get_env("OPENAI_API_KEY", required=True),
                    base_url=get_env("OPENAI_BASE_URL"),
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case ModelType.BEDROCK:
                return cls(
                    model_type=model_type,
                    api_key=get_env("BEDROCK_API_KEY", required=True),
                    base_url=get_env("BEDROCK_BASE_URL"),
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            case _:
                raise ValueError(f"Unsupported model type: {model_type}")


def get_or_create_ai_model(model_config: ModelConfig | None) -> Model:
    """Factory method for creating an AI model that is ready to use according to required configuration.

    Args:
        model_config (ModelConfig | None): The model configuration to use. If None, will create from environment.

    Returns:
        Model: The created AI model.

    Raises:
        ValueError: If the model configuration is invalid.
    """
    if model_config is None:
        model_config = ModelConfig.from_config()

    client_args: dict[str, Any] = {"api_key": model_config.api_key}
    if model_config.base_url:
        client_args["base_url"] = model_config.base_url

    match model_config.model_type:
        case ModelType.OPENAI:
            logger.info(f"Creating OpenAI model with ID: {model_config.model_id}")
            from strands.models.openai import OpenAIModel
            return OpenAIModel(
                client_args=client_args,
                model_id=model_config.model_id,
                params={
                    "max_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                },
            )
        case ModelType.LITELLM:
            logger.info(f"Creating LiteLLM model with ID: {model_config.model_id}")
            from strands.models.litellm import LiteLLMModel
            return LiteLLMModel(
                client_args=client_args,
                model_id=model_config.model_id,
                params={
                    "max_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                },
            )
        case ModelType.BEDROCK:
            logger.info(f"Creating Bedrock model with ID: {model_config.model_id}")
            from strands.models.bedrock import BedrockModel
            return BedrockModel(
                client_args=client_args,
                model_id=model_config.model_id,
                params={
                    "max_tokens": model_config.max_tokens,
                    "temperature": model_config.temperature,
                },
            )
        case _:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
