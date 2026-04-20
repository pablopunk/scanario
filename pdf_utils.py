"""PDF creation utilities for scanario."""

from pathlib import Path
from typing import List, Union

import img2pdf
from PIL import Image


def create_pdf_from_images(image_paths: List[Path], output_path: Path, dpi: int = 300) -> Path:
    """Create a PDF from a list of image paths.
    
    Args:
        image_paths: List of paths to images (will be converted if needed)
        output_path: Where to save the PDF
        dpi: DPI for the PDF
        
    Returns:
        Path to the created PDF
    """
    if not image_paths:
        raise ValueError("No images provided")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert any non-JPEG images to temporary JPEGs for img2pdf
    temp_paths = []
    try:
        for img_path in image_paths:
            if img_path.suffix.lower() in ['.jpg', '.jpeg']:
                temp_paths.append(str(img_path))
            else:
                # Convert to temporary JPEG
                temp_jpeg = output_path.parent / f"_temp_{img_path.stem}.jpg"
                with Image.open(img_path) as img:
                    if img.mode in ('RGBA', 'LA', 'P'):
                        img = img.convert('RGB')
                    img.save(temp_jpeg, 'JPEG', quality=95)
                temp_paths.append(str(temp_jpeg))
        
        # Create PDF
        with open(output_path, 'wb') as f:
            f.write(img2pdf.convert(temp_paths, dpi=dpi))
        
        return output_path
        
    finally:
        # Cleanup temporary files
        for temp_path in temp_paths:
            p = Path(temp_path)
            if p.name.startswith('_temp_'):
                p.unlink(missing_ok=True)


def collect_pages(
    new_images: List[Path] = None,
    existing_results: List[Path] = None,
    page_order: List[str] = None
) -> List[Path]:
    """Collect and order page images from various sources.
    
    Args:
        new_images: Paths to newly processed images
        existing_results: Paths to existing result directories or files
        page_order: Order specifier like ["new:0", "existing:job123", "new:1"]
                   If None, uses new_images first, then existing_results
    
    Returns:
        Ordered list of image paths for PDF creation
    """
    pages = []
    
    if page_order:
        for spec in page_order:
            source, idx = spec.split(':')
            idx = int(idx)
            if source == 'new':
                if new_images and idx < len(new_images):
                    pages.append(new_images[idx])
            elif source == 'existing':
                # idx refers to existing_results list index
                if existing_results and idx < len(existing_results):
                    result = existing_results[idx]
                    if result.is_dir():
                        # Get the enhanced image from this job
                        enhanced = list(result.glob('03-enhanced-*.jpg'))
                        if enhanced:
                            pages.append(enhanced[0])
                    elif result.is_file():
                        pages.append(result)
    else:
        # Default order: all new images, then best image from each existing result
        if new_images:
            pages.extend(new_images)
        
        if existing_results:
            for result in existing_results:
                if result.is_dir():
                    # Prefer enhanced image
                    enhanced = list(result.glob('03-enhanced-*.jpg'))
                    if enhanced:
                        pages.append(enhanced[0])
                    else:
                        # Fall back to any image
                        images = list(result.glob('*.jpg')) + list(result.glob('*.png'))
                        if images:
                            pages.append(images[0])
                elif result.is_file():
                    pages.append(result)
    
    return [p for p in pages if p and p.exists()]
