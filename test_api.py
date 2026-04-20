#!/usr/bin/env python3
"""Simple test script for the Scanario API."""

import argparse
import sys
import time
from pathlib import Path

import requests


def wait_for_job(base_url: str, job_id: str, timeout: int = 120) -> dict:
    """Poll job status until complete or timeout."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(f"{base_url}/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        
        if data["status"] == "completed":
            return data
        if data["status"] == "failed":
            print(f"Job failed: {data.get('error', 'Unknown error')}")
            sys.exit(1)
        
        print(f"  Status: {data['status']}...")
        time.sleep(2)
    
    print("Timeout waiting for job")
    sys.exit(1)


def test_scan(args):
    """Test the scan endpoint."""
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Upload
    print(f"Uploading {image_path.name}...")
    print(f"  mode={args.mode}, backend={args.backend}, debug={args.debug}")
    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        data = {"mode": args.mode, "backend": args.backend, "debug": str(args.debug).lower()}
        resp = requests.post(f"{args.url}/scan", files=files, data=data)
    
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    print(f"Job created: {job_id}")
    
    # Wait for completion
    print("Waiting for processing...")
    result = wait_for_job(args.url, job_id)
    
    print(f"\nCompleted! Generated files:")
    for f in result["files"]:
        print(f"  - {f}")
    
    # Download results
    if args.download:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading to {out_dir}/...")
        
        for filename in result["files"]:
            url = f"{args.url}/images/{job_id}/{filename}"
            resp = requests.get(url)
            resp.raise_for_status()
            
            out_path = out_dir / f"{job_id}_{filename}"
            out_path.write_bytes(resp.content)
            print(f"  Saved: {out_path}")
    
    print(f"\nImage URLs:")
    for filename in result["files"]:
        print(f"  {args.url}/images/{job_id}/{filename}")
    
    return job_id


def test_pdf(args):
    """Test the PDF endpoint."""
    files_to_upload = []
    existing_jobs = args.existing_jobs or []
    
    # Validate files
    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"Image not found: {p}")
            sys.exit(1)
        files_to_upload.append(p)
    
    print(f"Creating PDF with {len(files_to_upload)} new images and {len(existing_jobs)} existing jobs...")
    
    # Build request
    files = []
    for p in files_to_upload:
        files.append(("files", (p.name, open(p, "rb"), "image/jpeg")))
    
    data = {
        "mode": args.mode,
        "backend": args.backend,
        "debug": str(args.debug).lower(),
    }
    
    for job_id in existing_jobs:
        data.setdefault("existing_job_ids", []).append(job_id)
    
    # Add page order if specified
    if args.page_order:
        for spec in args.page_order:
            data.setdefault("page_order", []).append(spec)
    
    try:
        resp = requests.post(f"{args.url}/pdf", files=files, data=data)
        resp.raise_for_status()
    finally:
        for _, (_, f, _) in files:
            f.close()
    
    job = resp.json()
    job_id = job["job_id"]
    print(f"PDF job created: {job_id}")
    print(f"  Pages: {job['pages']}")
    
    # Wait for completion
    print("Waiting for PDF creation...")
    result = wait_for_job(args.url, job_id)
    
    print(f"\nCompleted! Generated files:")
    for f in result["files"]:
        print(f"  - {f}")
    
    # Download PDF
    if args.download or args.download_pdf:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_url = f"{args.url}/images/{job_id}/output.pdf"
        resp = requests.get(pdf_url)
        if resp.status_code == 200:
            out_path = out_dir / f"{job_id}_output.pdf"
            out_path.write_bytes(resp.content)
            print(f"\nPDF saved: {out_path}")
        else:
            print(f"PDF not found at {pdf_url}")
    
    print(f"\nPDF URL: {args.url}/images/{job_id}/output.pdf")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Test Scanario API")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Test single document scan")
    scan_parser.add_argument("image", help="Path to image file")
    scan_parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    scan_parser.add_argument("--mode", default="gray", choices=["gray", "archive", "color"], help="Enhancement mode")
    scan_parser.add_argument("--backend", default="auto", choices=["auto", "nano", "rembg"], help="Processing backend")
    scan_parser.add_argument("--debug", action="store_true", help="Save debug images")
    scan_parser.add_argument("--download", action="store_true", help="Download results")
    scan_parser.add_argument("--output-dir", default="./api_results", help="Download directory")
    
    # PDF command
    pdf_parser = subparsers.add_parser("pdf", help="Test PDF creation")
    pdf_parser.add_argument("images", nargs="*", help="Image files to include")
    pdf_parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    pdf_parser.add_argument("--mode", default="gray", choices=["gray", "archive", "color"], help="Enhancement mode")
    pdf_parser.add_argument("--backend", default="auto", choices=["auto", "nano", "rembg"], help="Processing backend")
    pdf_parser.add_argument("--debug", action="store_true", help="Save debug images")
    pdf_parser.add_argument("--existing-jobs", nargs="+", help="Existing job IDs to include")
    pdf_parser.add_argument("--page-order", nargs="+", help="Page ordering like file:0 job:abc123 file:1")
    pdf_parser.add_argument("--download", action="store_true", help="Download all results")
    pdf_parser.add_argument("--download-pdf", action="store_true", help="Download PDF only")
    pdf_parser.add_argument("--output-dir", default="./api_results", help="Download directory")
    
    args = parser.parse_args()
    
    if args.command == "scan":
        test_scan(args)
    elif args.command == "pdf":
        test_pdf(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
