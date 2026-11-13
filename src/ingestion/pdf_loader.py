"""MinerU PDF Loader.

Wrapper for the Magic-PDF (MinerU) CLI tool to convert PDF documents
into Markdown format suitable for ingestion.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import logging
from pathlib import Path

# Use local logger if setup_logging is complex to import or just use standard logging
logger = logging.getLogger(__name__)

class MinerULoader:
    """Wrapper for the Magic-PDF (MinerU) CLI.
    
    Converts PDF to Markdown for ingestion.
    """
    
    def __init__(self, output_dir: str = "data/mineru_output") -> None:
        """Initialize the loader.
        
        Args:
            output_dir: Directory to store intermediate MinerU outputs.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def parse_pdf(self, pdf_path: str) -> str:
        """Runs magic-pdf to convert PDF -> Markdown.
        
        Args:
            pdf_path: Path to the source PDF file.
            
        Returns:
            str: The content of the generated Markdown file.
            
        Raises:
            FileNotFoundError: If PDF does not exist.
            RuntimeError: If MinerU processing fails.
        """
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
            
        logger.info("🚀 Starting MinerU processing for: %s", pdf_file.name)
        
        # Check if magic-pdf is available
        if shutil.which("magic-pdf") is None:
             raise RuntimeError("magic-pdf command not found. Please install magic-pdf first.")

        # Construct MinerU CLI command
        # magic-pdf -p {file} -o {output_dir} -m auto
        # Note: magic-pdf creates a subdirectory with the PDF filename in the output dir
        cmd = [
            "magic-pdf",
            "-p", str(pdf_file.absolute()),
            "-o", str(self.output_dir.absolute()),
            "-m", "auto"
        ]
        
        try:
            # Run the heavy lifting
            result = subprocess.run(
                cmd, 
                check=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8', 
                errors='replace' # handle potential encoding issues in stdout
            )
            logger.debug("MinerU Output: %s", result.stdout)
            
            # Locate the generated markdown
            # MinerU creates a subdir named after the PDF stem in the output directory
            doc_dir = self.output_dir / pdf_file.stem
            
            # Check for generic auto-generated markdown naming usually {stem}.md
            md_files = list(doc_dir.glob("*.md"))
            
            # Sometimes it might name it 'auto.md' or similar depending on version; 
            # usually it's {filename}.md or just one .md file in that folder.
            if not md_files:
                 # Fallback search recursively in case structure changes
                 md_files = list(doc_dir.rglob("*.md"))

            if not md_files:
                logger.error("MinerU stdout: %s", result.stdout)
                logger.error("MinerU stderr: %s", result.stderr)
                raise ValueError(f"MinerU finished but no Markdown file found in {doc_dir}")
            
            # Pick the largest one if multiple (assuming implies content) or just the first
            target_md = md_files[0]
            logger.info("MinerU generation successful: %s", target_md)
                
            return target_md.read_text(encoding='utf-8')
            
        except subprocess.CalledProcessError as e:
            logger.error("❌ MinerU Failed with return code %d", e.returncode)
            logger.error("Stderr: %s", e.stderr)
            raise RuntimeError(f"PDF Parsing Failed: {e.stderr}")
