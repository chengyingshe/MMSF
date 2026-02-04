#!/usr/bin/env python3
"""
WSI Thumbnail Generator

This script generates thumbnails from Whole Slide Images (WSI) files.
It supports multiple WSI formats and can process files in batch.
"""

import os
import argparse
import time
import pandas as pd
from pathlib import Path
from PIL import Image, ImageDraw
import openslide
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET


def parse_annotation_xml(xml_path):
    """
    Parse XML annotation file to extract tumor regions
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        list: List of polygon coordinates for tumor regions
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        annotations = []
        
        # Find all annotations that are part of "Tumor" group
        for annotation in root.findall('.//Annotation'):
            group = annotation.get('PartOfGroup')
            if group == 'Tumor':
                coordinates = []
                for coord in annotation.findall('.//Coordinate'):
                    x = float(coord.get('X'))
                    y = float(coord.get('Y'))
                    coordinates.append((x, y))
                
                if coordinates:
                    annotations.append(coordinates)
        
        return annotations
    except Exception as e:
        print(f"Error parsing annotation file {xml_path}: {e}")
        return []


def get_wsi_info(wsi_path):
    """
    Get basic information about a WSI file
    
    Args:
        wsi_path: Path to WSI file
        
    Returns:
        dict: WSI information including dimensions, levels, etc.
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        info = {
            'levels': slide.level_count,
            'dimensions': slide.dimensions,
            'level_dimensions': slide.level_dimensions,
            'level_downsamples': slide.level_downsamples,
            'properties': dict(slide.properties)
        }
        slide.close()
        return info
    except Exception as e:
        print(f"Error reading {wsi_path}: {e}")
        return None


def create_thumbnail(wsi_path, output_path, target_size=(1024, 1024), level=None, annotation_path=None):
    """
    Create thumbnail from WSI file with optional tumor annotations
    
    Args:
        wsi_path: Path to WSI file
        output_path: Path to save thumbnail
        target_size: Target thumbnail size (width, height) - minimum edge will be scaled to this size
        level: Specific level to use for thumbnail (None for auto)
        annotation_path: Path to XML annotation file (optional)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        
        # Determine best level for thumbnail
        if level is None:
            # Find the best level that gives us a thumbnail close to target_size
            best_level = 0
            min_target_size = min(target_size[0], target_size[1])
            for i in range(slide.level_count):
                level_dim = slide.level_dimensions[i]
                min_level_size = min(level_dim[0], level_dim[1])
                if min_level_size <= min_target_size * 2:
                    best_level = i
                    break
        else:
            best_level = min(level, slide.level_count - 1)
        
        # Get the level dimensions
        level_dim = slide.level_dimensions[best_level]
        
        # Read the thumbnail
        thumbnail = slide.read_region((0, 0), best_level, level_dim)
        thumbnail = thumbnail.convert('RGB')
        
        # Calculate resize dimensions maintaining aspect ratio
        # Scale so that the minimum edge equals the minimum target size
        original_width, original_height = thumbnail.size
        min_target_size = min(target_size[0], target_size[1])
        
        # Calculate scale factor based on minimum edge
        if original_width <= original_height:
            # Width is the minimum edge
            scale_factor = min_target_size / original_width
            new_width = min_target_size
            new_height = int(original_height * scale_factor)
        else:
            # Height is the minimum edge
            scale_factor = min_target_size / original_height
            new_width = int(original_width * scale_factor)
            new_height = min_target_size
        
        # Resize maintaining aspect ratio
        if thumbnail.size != (new_width, new_height):
            thumbnail = thumbnail.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Add annotations if provided
        if annotation_path and os.path.exists(annotation_path):
            # Parse annotations
            tumor_regions = parse_annotation_xml(annotation_path)
            
            if tumor_regions:
                # Create a drawing object
                draw = ImageDraw.Draw(thumbnail)
                
                # Get original WSI dimensions for scaling
                original_wsi_width, original_wsi_height = slide.dimensions
                
                # Calculate scaling factors
                wsi_scale_x = new_width / original_wsi_width
                wsi_scale_y = new_height / original_wsi_height
                
                # Draw tumor regions
                for region in tumor_regions:
                    # Scale coordinates to thumbnail size
                    scaled_coords = []
                    for x, y in region:
                        scaled_x = int(x * wsi_scale_x)
                        scaled_y = int(y * wsi_scale_y)
                        scaled_coords.append((scaled_x, scaled_y))
                    
                    # Draw polygon with red outline and semi-transparent red fill
                    if len(scaled_coords) >= 3:
                        # Create a separate image for the polygon to enable transparency
                        polygon_img = Image.new('RGBA', thumbnail.size, (0, 0, 0, 0))
                        polygon_draw = ImageDraw.Draw(polygon_img)
                        polygon_draw.polygon(scaled_coords, fill=(255, 0, 0, 64), outline=(255, 0, 0, 255))
                        
                        # Composite the polygon onto the thumbnail
                        thumbnail = Image.alpha_composite(thumbnail.convert('RGBA'), polygon_img).convert('RGB')
        
        # Save thumbnail
        thumbnail.save(output_path, 'JPEG', quality=95)
        slide.close()
        
        return True
        
    except Exception as e:
        print(f"Error creating thumbnail for {wsi_path}: {e}")
        return False


def process_single_wsi(wsi_path, output_dir, target_size=(1024, 1024), level=None, annotation_dir=None):
    """
    Process a single WSI file to create thumbnail
    
    Args:
        wsi_path: Path to WSI file
        output_dir: Directory to save thumbnail
        target_size: Target thumbnail size - minimum edge will be scaled to this size
        level: Specific level to use
        annotation_dir: Directory containing annotation files (optional)
        
    Returns:
        dict: Processing result
    """
    start_time = time.time()
    
    # Get file info
    slide_id = Path(wsi_path).stem
    output_path = os.path.join(output_dir, f"{slide_id}.jpg")
    
    # Check if thumbnail already exists
    if os.path.exists(output_path):
        print(f"Thumbnail already exists: {output_path}")
        return {
            'slide_id': slide_id,
            'status': 'already_exists',
            'time': 0,
            'output_path': output_path
        }
    
    # Get WSI info
    wsi_info = get_wsi_info(wsi_path)
    if wsi_info is None:
        return {
            'slide_id': slide_id,
            'status': 'failed_info',
            'time': time.time() - start_time,
            'output_path': None
        }
    
    # Find corresponding annotation file
    annotation_path = None
    if annotation_dir:
        annotation_file = os.path.join(annotation_dir, f"{slide_id}.xml")
        if os.path.exists(annotation_file):
            annotation_path = annotation_file
            print(f"Found annotation file: {annotation_file}")
        else:
            print(f"No annotation file found for {slide_id}")
    
    # Create thumbnail
    success = create_thumbnail(wsi_path, output_path, target_size, level, annotation_path)
    
    processing_time = time.time() - start_time
    
    if success:
        return {
            'slide_id': slide_id,
            'status': 'success',
            'time': processing_time,
            'output_path': output_path,
            'wsi_info': wsi_info,
            'has_annotations': annotation_path is not None
        }
    else:
        return {
            'slide_id': slide_id,
            'status': 'failed',
            'time': processing_time,
            'output_path': None
        }


def find_wsi_files(source_dir, wsi_format="svs"):
    """
    Find all WSI files in the source directory
    
    Args:
        source_dir: Source directory to search
        wsi_format: WSI file format(s) to search for
        
    Returns:
        list: List of WSI file paths
    """
    slides = []
    
    # Handle multiple formats
    if ';' in wsi_format:
        wsi_formats = wsi_format.split(";")
    else:
        wsi_formats = [wsi_format]
    
    # Search for WSI files
    for root, dirs, filenames in os.walk(source_dir):
        for filename in filenames:
            postfix = filename.split(".")[-1].lower()
            if postfix in wsi_formats:
                slides.append(os.path.join(root, filename))
    
    return slides


def create_thumbnails_batch(
    source_dir,
    output_dir,
    target_size=(1024, 1024),
    level=None,
    wsi_format="svs",
    auto_skip=True,
    process_list=None,
    annotation_dir=None
):
    """
    Create thumbnails for multiple WSI files
    
    Args:
        source_dir: Source directory containing WSI files
        output_dir: Output directory for thumbnails
        target_size: Target thumbnail size - minimum edge will be scaled to this size
        level: Specific level to use
        wsi_format: WSI file format
        auto_skip: Skip if thumbnail already exists
        process_list: CSV file with specific files to process
        annotation_dir: Directory containing annotation files (optional)
        
    Returns:
        dict: Processing statistics
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find WSI files
    if process_list and os.path.exists(process_list):
        df = pd.read_csv(process_list)
        slides = df['slide_id'].tolist()
        # Convert to full paths
        slides = [os.path.join(source_dir, slide) for slide in slides]
    else:
        slides = find_wsi_files(source_dir, wsi_format)
    
    print(f"Found {len(slides)} WSI files to process")
    
    # Process each slide
    results = []
    total_time = 0
    annotated_count = 0
    
    for i, slide_path in enumerate(tqdm(slides, desc="Creating thumbnails")):
        print(f"\nProcessing {i+1}/{len(slides)}: {os.path.basename(slide_path)}")
        
        result = process_single_wsi(slide_path, output_dir, target_size, level, annotation_dir)
        results.append(result)
        total_time += result['time']
        
        if result.get('has_annotations', False):
            annotated_count += 1
        
        print(f"Status: {result['status']}, Time: {result['time']:.2f}s")
    
    # Create summary
    status_counts = {}
    for result in results:
        status = result['status']
        status_counts[status] = status_counts.get(status, 0) + 1
    
    summary = {
        'total_files': len(slides),
        'successful': status_counts.get('success', 0),
        'already_exists': status_counts.get('already_exists', 0),
        'failed': status_counts.get('failed', 0) + status_counts.get('failed_info', 0),
        'annotated': annotated_count,
        'total_time': total_time,
        'average_time': total_time / len(slides) if slides else 0,
        'results': results
    }
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(output_dir, 'thumbnail_results.csv')
    results_df.to_csv(results_csv, index=False)
    
    print(f"\nSummary:")
    print(f"Total files: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Already existed: {summary['already_exists']}")
    print(f"Failed: {summary['failed']}")
    print(f"With annotations: {summary['annotated']}")
    print(f"Total time: {summary['total_time']:.2f}s")
    print(f"Average time per file: {summary['average_time']:.2f}s")
    print(f"Results saved to: {results_csv}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Create thumbnails from WSI files")
    parser.add_argument("--source", type=str, default='sample_wsi/wsi/', 
                       help="Path to folder containing WSI files")
    parser.add_argument("--output", type=str, default='sample_wsi/thumbnails',
                       help="Directory to save thumbnails")
    parser.add_argument("--target_size", type=int, nargs=2, default=[2048, 2048],
                       help="Target thumbnail size (width height) - minimum edge will be scaled to this size")
    parser.add_argument("--level", type=int, default=None,
                       help="Specific WSI level to use for thumbnail")
    parser.add_argument("--wsi_format", type=str, default="svs;tif;ndpi",
                       help="WSI file format(s), use semicolon to separate multiple formats")
    parser.add_argument("--no_auto_skip", action="store_true",
                       help="Don't skip existing thumbnails")
    parser.add_argument("--process_list", type=str, default=None,
                       help="CSV file with specific files to process")
    parser.add_argument("--annotation_dir", type=str, default=None,
                       help="Directory containing XML annotation files")
    
    args = parser.parse_args()
    
    print("WSI Thumbnail Generator")
    print("=" * 50)
    print(f"Source directory: {args.source}")
    print(f"Output directory: {args.output}")
    print(f"Target size: {args.target_size}")
    print(f"WSI format: {args.wsi_format}")
    print(f"Auto skip: {not args.no_auto_skip}")
    if args.annotation_dir:
        print(f"Annotation directory: {args.annotation_dir}")
    print("=" * 50)
    
    # Create thumbnails
    summary = create_thumbnails_batch(
        source_dir=args.source,
        output_dir=args.output,
        target_size=tuple(args.target_size),
        level=args.level,
        wsi_format=args.wsi_format,
        auto_skip=not args.no_auto_skip,
        process_list=args.process_list,
        annotation_dir=args.annotation_dir
    )
    
    print("\nThumbnail generation completed!")


if __name__ == "__main__":
    main() 