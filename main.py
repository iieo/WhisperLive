#!/usr/bin/env python3
"""
Run the WhisperLive server with Parakeet v3 support

Usage:
    python run_server.py [options]
    
Example:
    python run_server.py --backend parakeet --port 8005
"""

from whisper_live.server import TranscriptionServer
import sys
import os
import argparse
import logging

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the server after path is set

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description='WhisperLive Server with Parakeet v3 Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with Faster Whisper (default)
  python run_server.py
  
  # Run with Parakeet v3
  python run_server.py --backend parakeet
  
  # Run with custom Parakeet model
  python run_server.py --backend parakeet --parakeet-model nvidia/parakeet-tdt-1.1b
  
  # Run with single model mode for better performance
  python run_server.py --backend parakeet --single-model
        """
    )

    # Server configuration
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host address to bind the server (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8005,
                        help='Port number to bind the server (default: 8005)')

    # Backend selection
    parser.add_argument('--backend', type=str, default='parakeet',
                        choices=['faster_whisper', 'parakeet',
                                 'tensorrt', 'openvino'],
                        help='Backend to use for transcription (default: parakeet)')

    # Model configuration
    parser.add_argument('--faster-whisper-model', type=str, default='small.en',
                        help='Faster Whisper model name')
    parser.add_argument('--tensorrt-model-path', type=str,
                        help='Path to TensorRT model')

    # Performance options
    parser.add_argument('--single-model', action='store_true',
                        help='Use single model instance for all clients (saves memory)')
    parser.add_argument('--max-clients', type=int, default=4,
                        help='Maximum number of concurrent clients (default: 4)')
    parser.add_argument('--max-connection-time', type=int, default=600,
                        help='Maximum connection time per client in seconds (default: 600)')

    # Cache directory
    parser.add_argument('--cache-dir', type=str, default='~/.cache/whisper-live/',
                        help='Cache directory for models (default: ~/.cache/whisper-live/)')

    # Logging
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Logging level (default: INFO)')

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Print configuration
    logging.info("=" * 60)
    logging.info("WhisperLive Server with Parakeet v3 Support")
    logging.info("=" * 60)
    logging.info(f"Configuration:")
    logging.info(f"  Host: {args.host}")
    logging.info(f"  Port: {args.port}")
    logging.info(f"  Backend: {args.backend}")
    if args.backend == 'faster_whisper':
        logging.info(f"  Model: {args.faster_whisper_model}")
    logging.info(f"  Single Model Mode: {args.single_model}")
    logging.info(f"  Max Clients: {args.max_clients}")
    logging.info(f"  Max Connection Time: {args.max_connection_time}s")
    logging.info("=" * 60)

    try:
        # Create server instance
        server = TranscriptionServer()

        # Determine model path based on backend
        faster_whisper_model = None
        if args.backend == 'faster_whisper' and args.faster_whisper_model != 'small.en':
            faster_whisper_model = args.faster_whisper_model

        # Run the server
        logging.info(f"Starting server at ws://{args.host}:{args.port}")
        logging.info("Press Ctrl+C to stop the server")

        server.run(
            host=args.host,
            port=args.port,
            single_model=args.single_model,
            max_clients=args.max_clients,
            max_connection_time=args.max_connection_time,
            cache_path=args.cache_dir
        )

    except KeyboardInterrupt:
        logging.info("\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
