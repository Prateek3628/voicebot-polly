#!/usr/bin/env python3
"""
Script to recover from ChromaDB corruption on production servers.
Run this when you see Rust panic errors or database corruption issues.
"""
import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def backup_corrupted_database():
    """Backup the corrupted ChromaDB database."""
    db_path = Path(config.chromadb_persist_directory)
    
    if not db_path.exists():
        logger.warning(f"Database path does not exist: {db_path}")
        return None
    
    # Create backup with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.name}.backup.{timestamp}"
    
    try:
        logger.info(f"📦 Backing up corrupted database...")
        logger.info(f"   From: {db_path}")
        logger.info(f"   To: {backup_path}")
        
        shutil.move(str(db_path), str(backup_path))
        logger.info(f"✅ Backup created successfully")
        return backup_path
    except Exception as e:
        logger.error(f"❌ Failed to backup database: {e}")
        raise


def create_fresh_database():
    """Create a fresh ChromaDB directory."""
    db_path = Path(config.chromadb_persist_directory)
    
    try:
        logger.info(f"🔨 Creating fresh database directory...")
        db_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Fresh database directory created: {db_path}")
    except Exception as e:
        logger.error(f"❌ Failed to create database directory: {e}")
        raise


def reinitialize_data():
    """Reinitialize the database with data."""
    try:
        logger.info(f"📝 Reinitializing database with data...")
        
        # Import here to avoid circular dependencies
        from scripts.initialise_data import main as init_main
        
        init_main()
        logger.info(f"✅ Database reinitialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to reinitialize database: {e}")
        logger.info(f"💡 You can manually run: python scripts/initialise_data.py")
        raise


def main():
    """Main recovery function."""
    logger.info("=" * 60)
    logger.info("🔧 ChromaDB Recovery Tool")
    logger.info("=" * 60)
    
    try:
        # Step 1: Backup corrupted database
        backup_path = backup_corrupted_database()
        if backup_path:
            logger.info(f"📁 Old database backed up to: {backup_path}")
        
        # Step 2: Create fresh database
        create_fresh_database()
        
        # Step 3: Ask user if they want to reinitialize data
        response = input("\n🔄 Do you want to reinitialize the database with data now? (y/n): ").strip().lower()
        
        if response == 'y':
            reinitialize_data()
        else:
            logger.info("⚠️  Database created but empty. Run 'python scripts/initialise_data.py' to load data.")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ Recovery completed successfully!")
        logger.info("=" * 60)
        logger.info("\n💡 Next steps:")
        logger.info("   1. Restart your server")
        logger.info("   2. Test the connection")
        logger.info("   3. If issues persist, check disk space and permissions")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Recovery cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n❌ Recovery failed: {e}")
        logger.info("\n💡 Manual recovery steps:")
        logger.info("   1. Stop the server")
        logger.info(f"   2. Backup: mv {config.chromadb_persist_directory} {config.chromadb_persist_directory}.backup")
        logger.info(f"   3. Recreate: mkdir -p {config.chromadb_persist_directory}")
        logger.info("   4. Run: python scripts/initialise_data.py")
        logger.info("   5. Restart the server")
        sys.exit(1)


if __name__ == "__main__":
    main()
