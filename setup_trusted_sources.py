"""
Setup script to initialize trusted sources for the RAGS system.
Run this script to register all documents in your docs directory as trusted sources.
"""

from utils.sourceValidator import SourceValidator

def main():
    """Initialize trusted sources whitelist."""
    print("🔒 Setting up RAGS Source Security...")

    # Initialize validator
    validator = SourceValidator()

    # Auto-register all docs
    registered_count = validator.auto_register_docs_directory()

    if registered_count > 0:
        # Print summary
        summary = validator.get_whitelist_summary()
        print(f"\n📊 Security Setup Complete!")
        print(f"✅ Total trusted sources: {summary['total_sources']}")
        print(f"📁 Whitelist file: {summary['sources'][0] if summary['sources'] else 'None'}")

        print("\n🔐 Registered Sources:")
        for source in summary['sources']:
            print(f"  - {source['file_name']} (registered: {source['registered_date'][:10]})")
    else:
        print("❌ No documents found to register. Please add PDF files to the 'docs' directory.")

    print(f"\n🛡️ Source validation is now active. Only registered documents will be processed.")

if __name__ == "__main__":
    main()