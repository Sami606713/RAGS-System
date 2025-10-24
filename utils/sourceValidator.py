"""
Source Validation Module for RAGS Pipeline

This module ensures only trusted, uploaded sources are processed and retrieved.
It maintains a whitelist of valid sources and validates all document operations.
"""

import os
import json
import hashlib
from typing import List, Set, Dict, Optional
from datetime import datetime
from pathlib import Path


class SourceValidator:
    """
    Validates document sources to prevent untrusted content from entering the RAG pipeline.
    Maintains a whitelist of approved sources and their metadata.
    """

    def __init__(self, whitelist_path: str = "trusted_sources.json", docs_directory: str = "docs"):
        self.whitelist_path = whitelist_path
        self.docs_directory = docs_directory
        self.trusted_sources = self._load_whitelist()

    def _load_whitelist(self) -> Dict[str, Dict]:
        """Load trusted sources from whitelist file."""
        if os.path.exists(self.whitelist_path):
            try:
                with open(self.whitelist_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load whitelist: {e}")
                return {}
        return {}

    def _save_whitelist(self):
        """Save trusted sources to whitelist file."""
        try:
            with open(self.whitelist_path, 'w') as f:
                json.dump(self.trusted_sources, f, indent=2)
        except Exception as e:
            print(f"Error saving whitelist: {e}")

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate SHA256 hash for file integrity verification."""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error generating hash for {file_path}: {e}")
            return ""

    def register_trusted_source(self, file_path: str, description: str = "") -> bool:
        """
        Register a new trusted source in the whitelist.

        Args:
            file_path: Path to the trusted document
            description: Optional description of the source

        Returns:
            bool: True if successfully registered
        """
        try:
            if not os.path.exists(file_path):
                print(f"Error: File {file_path} does not exist")
                return False

            abs_path = os.path.abspath(file_path)
            file_hash = self._generate_file_hash(abs_path)

            if not file_hash:
                return False

            # Normalize path for consistent storage
            normalized_path = abs_path.replace("\\", "/")

            source_info = {
                "file_path": normalized_path,
                "file_hash": file_hash,
                "file_size": os.path.getsize(abs_path),
                "registered_date": datetime.now().isoformat(),
                "description": description,
                "file_name": os.path.basename(abs_path)
            }

            self.trusted_sources[normalized_path] = source_info
            self._save_whitelist()

            print(f">> Registered trusted source: {os.path.basename(abs_path)}")
            return True

        except Exception as e:
            print(f"Error registering source {file_path}: {e}")
            return False

    def is_trusted_source(self, file_path: str) -> bool:
        """
        Check if a file path corresponds to a trusted source.

        Args:
            file_path: Path to check

        Returns:
            bool: True if source is trusted
        """
        if not file_path:
            return False

        # Normalize path for comparison
        normalized_path = os.path.abspath(file_path).replace("\\", "/")

        # Check if path is in whitelist
        if normalized_path in self.trusted_sources:
            # Verify file integrity if file still exists
            if os.path.exists(file_path):
                current_hash = self._generate_file_hash(file_path)
                stored_hash = self.trusted_sources[normalized_path].get("file_hash", "")

                if current_hash == stored_hash:
                    return True
                else:
                    print(f"WARNING: File hash mismatch for {os.path.basename(file_path)}")
                    return False
            else:
                print(f"WARNING: Trusted file no longer exists: {os.path.basename(file_path)}")
                return False

        return False

    def validate_document_metadata(self, metadata: Dict) -> bool:
        """
        Enhanced validation with strict enforcement and provenance tracking.

        Args:
            metadata: Document metadata dictionary

        Returns:
            bool: True if metadata is valid and from trusted source
        """
        # Strict source validation
        source_path = metadata.get('source', '')
        source_file = metadata.get('source_file', '')

        if not source_path:
            print("SECURITY: Document rejected - missing source metadata")
            return False

        # Cross-validate source consistency
        if source_file and not source_path.endswith(source_file):
            print(f"SECURITY: Source path mismatch - {source_path} vs {source_file}")
            return False

        # Verify trusted source with additional checks
        if not self.is_trusted_source(source_path):
            print(f"SECURITY: Untrusted source blocked - {source_path}")
            return False

        # Additional metadata integrity checks
        required_fields = ['chunk_id', 'type']
        missing_fields = [field for field in required_fields if field not in metadata]
        if missing_fields:
            print(f"SECURITY: Missing required metadata fields: {missing_fields}")
            return False

        return True

    def get_document_provenance(self, metadata: Dict) -> Dict:
        """Extract detailed provenance information for citations"""
        return {
            'source_file': metadata.get('source_file', 'Unknown'),
            'source_path': metadata.get('source', 'Unknown'),
            'chunk_id': metadata.get('chunk_id', 'Unknown'),
            'semantic_type': metadata.get('semantic_type', 'content'),
            'header': metadata.get('header', ''),
            'processed_date': metadata.get('processed_date', 'Unknown'),
            'is_trusted': self.validate_document_metadata(metadata)
        }

    def get_trusted_sources_list(self) -> List[str]:
        """Return list of all trusted source file paths."""
        return list(self.trusted_sources.keys())

    def remove_trusted_source(self, file_path: str) -> bool:
        """
        Remove a source from the trusted whitelist.

        Args:
            file_path: Path to remove from whitelist

        Returns:
            bool: True if successfully removed
        """
        normalized_path = os.path.abspath(file_path).replace("\\", "/")

        if normalized_path in self.trusted_sources:
            del self.trusted_sources[normalized_path]
            self._save_whitelist()
            print(f">> Removed trusted source: {os.path.basename(file_path)}")
            return True
        else:
            print(f"ERROR: Source not found in whitelist: {os.path.basename(file_path)}")
            return False

    def auto_register_docs_directory(self) -> int:
        """
        Automatically register all PDF and JSON files in the docs directory as trusted sources.

        Returns:
            int: Number of sources registered
        """
        if not os.path.exists(self.docs_directory):
            print(f"ERROR: Docs directory not found: {self.docs_directory}")
            return 0

        registered_count = 0

        for file_name in os.listdir(self.docs_directory):
            if file_name.lower().endswith(('.pdf', '.json')):
                file_path = os.path.join(self.docs_directory, file_name)
                if self.register_trusted_source(file_path, "Auto-registered from docs directory"):
                    registered_count += 1

        print(f">> Auto-registered {registered_count} documents from {self.docs_directory}")
        return registered_count

    def get_whitelist_summary(self) -> Dict:
        """Get summary statistics about the whitelist."""
        return {
            "total_sources": len(self.trusted_sources),
            "whitelist_file": self.whitelist_path,
            "sources": [
                {
                    "file_name": info["file_name"],
                    "registered_date": info["registered_date"],
                    "description": info["description"]
                }
                for info in self.trusted_sources.values()
            ]
        }


# Global instance
_source_validator = None

def get_source_validator() -> SourceValidator:
    """Get global SourceValidator instance."""
    global _source_validator
    if _source_validator is None:
        _source_validator = SourceValidator()
    return _source_validator


def validate_source(file_path: str) -> bool:
    """Quick validation function for external use."""
    return get_source_validator().is_trusted_source(file_path)


if __name__ == "__main__":
    # Example usage
    validator = SourceValidator()

    # Auto-register all docs
    validator.auto_register_docs_directory()

    # Print summary
    summary = validator.get_whitelist_summary()
    print(f"\n📊 Whitelist Summary:")
    print(f"Total trusted sources: {summary['total_sources']}")
    for source in summary['sources']:
        print(f"  - {source['file_name']} (registered: {source['registered_date'][:10]})")