#!/usr/bin/env python3
"""
Helper to load environment variables from .env file.
Usage: from load_env import load_env; load_env()
"""
from pathlib import Path
import os


def load_env(env_file: str = ".env") -> dict:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file (relative to project root)
    
    Returns:
        Dict of environment variables that were loaded
    """
    # Find project root
    current = Path(__file__).resolve().parent
    project_root = current.parent
    env_path = project_root / env_file
    
    if not env_path.exists():
        print(f"⚠️  No {env_file} file found at {env_path}")
        print(f"   Copy .env.example to .env and fill in your credentials")
        return {}
    
    loaded = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Set environment variable
                os.environ[key] = value
                loaded[key] = value
    
    print(f"✓ Loaded {len(loaded)} environment variables from {env_file}")
    return loaded


if __name__ == "__main__":
    loaded = load_env()
    
    # Show what was loaded (without exposing values)
    if loaded:
        print("\nLoaded variables:")
        for key in loaded:
            # Mask sensitive values
            if any(sensitive in key.lower() for sensitive in ["token", "password", "secret", "key"]):
                print(f"  {key}=***MASKED***")
            else:
                print(f"  {key}={loaded[key]}")
