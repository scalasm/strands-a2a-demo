#!/usr/bin/env python3
"""
AWS Credentials Helper

Utility to ensure AWS credentials are available in environment variables
for EnvironmentCredentialsResolver to use, regardless of where boto3 finds them.
"""

import os
import boto3
import logging
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

logger = logging.getLogger(__name__)


def _setup_aws_credentials_in_env():
    """
    Use boto3 to find AWS credentials from any source and set them in environment variables.
    
    This ensures EnvironmentCredentialsResolver can find them, even if boto3 got them
    from profiles, instance metadata, or other sources.
    
    Returns:
        bool: True if credentials were successfully set, False otherwise
    """
    try:
        # Create a boto3 session to get credentials from any available source
        # Use AWS_PROFILE if it's set in the environment
        profile_name = os.environ.get('AWS_PROFILE')
        if profile_name:
            logger.debug(f"Using AWS profile: {profile_name}")
            session = boto3.Session(profile_name=profile_name)
        else:
            session = boto3.Session()
        
        # Get credentials object
        credentials = session.get_credentials()
        
        if not credentials:
            logger.warning("No AWS credentials found by boto3")
            return False
        
        # Get the actual credential values
        access_key = credentials.access_key
        secret_key = credentials.secret_key
        session_token = credentials.token  # This might be None for non-temporary credentials
        
        if not access_key or not secret_key:
            logger.warning("AWS credentials are incomplete (missing access key or secret key)")
            return False
        
        # Set environment variables that EnvironmentCredentialsResolver expects
        os.environ['AWS_ACCESS_KEY_ID'] = access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = secret_key
        
        # Set session token if available (for temporary credentials)
        if session_token:
            os.environ['AWS_SESSION_TOKEN'] = session_token
            logger.debug("Set AWS credentials in environment (including session token)")
        else:
            # Remove session token if it exists but we don't have one
            # This prevents issues with stale session tokens
            if 'AWS_SESSION_TOKEN' in os.environ:
                del os.environ['AWS_SESSION_TOKEN']
            logger.debug("Set AWS credentials in environment (no session token)")
        
        return True
        
    except (NoCredentialsError, PartialCredentialsError) as e:
        logger.warning(f"AWS credentials not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Error setting up AWS credentials: {e}")
        return False


def ensure_aws_credentials():
    """
    Convenience function to ensure AWS credentials are available.
    
    Checks if credentials are already in environment variables first,
    and only uses boto3 to find them if they're not already set.
    
    Returns:
        bool: True if credentials are available, False otherwise
    """
    # Check if credentials are already in environment
    if (os.environ.get('AWS_ACCESS_KEY_ID') and 
        os.environ.get('AWS_SECRET_ACCESS_KEY')):
        logger.debug("AWS credentials already available in environment")
        return True
    
    # Try to set them using boto3
    logger.debug("AWS credentials not in environment, attempting to find them with boto3")
    return _setup_aws_credentials_in_env() 