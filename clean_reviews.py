import json
import re
import os
from urllib.parse import urlparse
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_review_text(text, url):
    """Cleans the messy review text."""
    if not isinstance(text, str) or not text.strip():
        logging.warning(f"Empty or non-string content found for URL: {url}")
        return ""

    # 1. Attempt Deduplication: Find the end of the main review content.
    # Common markers indicating the end of the review body / start of footer junk
    end_markers = [
        "Pros:",
        "Cons:",
        "Conclusion",
        "GEAR USED:",
        "Purchase the",
        "Keywords:",
        "Share on Facebook",
        "Want to support this channel?",
        "Buy DA Merchandise",
        "_________________________________________________________________________" # Long underscore line
    ]
    first_end_pos = len(text)
    # Look for the first occurrence of any end marker
    for marker in end_markers:
        try:
            # Case-insensitive search might be safer
            pos = text.lower().find(marker.lower())
            if pos != -1 and pos < first_end_pos:
                first_end_pos = pos
        except Exception as e:
            logging.error(f"Error finding marker '{marker}' in text for {url}: {e}")
            continue # Should not happen with find, but good practice

    # Take the text up to the first end marker found
    meaningful_text = text[:first_end_pos].strip()

    # If the text is still very short, the marker might have been too early.
    # As a fallback, split by common very large separators and take the first part.
    if len(meaningful_text) < 300: # Arbitrary threshold, adjust if needed
        logging.warning(f"Initial text cut short for {url}, attempting fallback split.")
        parts = re.split(r'\n\n[-_]{20,}\n\n', text, maxsplit=1)
        if len(parts) > 0 and len(parts[0]) > 300:
            meaningful_text = parts[0].strip()
        else:
             # If still short, log it but proceed with the initial short version
             logging.warning(f"Fallback split also resulted in short text for {url}. Check content manually.")


    cleaned = meaningful_text

    # 2. Noise Removal (Specific patterns from the example)
    # Remove social media/intro lines more robustly
    cleaned = re.sub(r'^Follow Me @.*?(\n|$)', '', cleaned, flags=re.MULTILINE | re.IGNORECASE).strip()
    cleaned = re.sub(r'^Thanks to .*? for sending me.*?(\n|$)', '', cleaned, flags=re.MULTILINE | re.IGNORECASE).strip()
    cleaned = re.sub(r'^\*The tests and most of the photos.*?(\n|$)', '', cleaned, flags=re.MULTILINE | re.IGNORECASE).strip()
    cleaned = re.sub(r'^You can visit the product page.*?(\n|$)', '', cleaned, flags=re.MULTILINE | re.IGNORECASE).strip()

    # Remove lists of other reviews
    cleaned = re.sub(r'Viltrox AIR Series Reviews:.*?\n(Viltrox AF.*?(\n|$))+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Here’s a look at my reviews of this series.*?(\n|$)', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)


    # Remove promotional codes more generally
    cleaned = re.sub(r'\(use code .*? for .*?% off\)', '', cleaned, flags=re.IGNORECASE)

    # Remove image/video references (making them less disruptive)
    cleaned = re.sub(r'watching the video review below\s*or reading on', 'as detailed below', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'For example, here is .*? at F\d+(\.\d)? compared to .*?:', '[Comparison description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'But here is something absurd: check out the corner comparison!', '[Corner comparison description follows.]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Oof!', '', cleaned) # Often follows an image comparison
    cleaned = re.sub(r'Here’s a deep crop from a photo.*?:', '[Deep crop description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Here’s an image taken at F\d+:', '[Image description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Here’s what that looks like:', '[Magnification example description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Here’s a grab from a video clip.*?:', '[Video frame description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'Here’s a look at my test chart:', '[Test chart description:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'And here are the crops.*?:', '[Crop descriptions follow:]', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'visit the image gallery here', '(image gallery link removed)', cleaned, flags=re.IGNORECASE)


    # 3. Formatting Cleanup
    cleaned = re.sub(r'_{5,}', '', cleaned) # Remove underscore lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned) # Consolidate multiple newlines

    return cleaned.strip()

def extract_title_from_url(url):
    """Attempts to create a title from the URL slug."""
    try:
        path = urlparse(url).path
        # Remove potential trailing date parts or file extensions if needed
        path = re.sub(r'/\d{4}/\d{2}/?$', '', path) # Remove /YYYY/MM/
        slug = os.path.basename(path.strip('/'))
        if slug:
            # Basic cleaning: replace hyphens/underscores, title case
            title = slug.replace('-', ' ').replace('_', ' ').title()
            # Remove common trailing words if they seem like categories/types
            common_endings = [" Review", " G Review", " Air Review"] # Add more if needed
            for ending in common_endings:
                if title.lower().endswith(ending.lower()):
                    title = title[:-len(ending)]
                    break # Stop after first match
            return title.strip()
    except Exception as e:
        logging.error(f"Error parsing title from URL {url}: {e}")
        pass
    return ""

def extract_date_from_url(url):
    """Attempts to extract YYYY-MM from URL path."""
    # Match patterns like /YYYY/MM/ or /YYYY/MM
    match = re.search(r'/(\d{4})/(\d{1,2})(/|$)', url)
    if match:
        year, month = match.groups()[:2]
        return f"{year}-{int(month):02d}" # Format as YYYY-MM
    return None

# --- Main Script ---
input_filename = 'dustin_photography_reviews.json'
output_filename = 'dustin_photography_reviews_cleaned.json'

logging.info(f"Starting cleaning process for '{input_filename}'...")

try:
    with open(input_filename, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input JSON is not a list of objects.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from '{input_filename}': {e}")
            exit(1)
        except ValueError as e:
             logging.error(f"JSON structure error in '{input_filename}': {e}")
             exit(1)

except FileNotFoundError:
    logging.error(f"Error: Input file '{input_filename}' not found.")
    exit(1)
except Exception as e:
     logging.error(f"Failed to open or read '{input_filename}': {e}")
     exit(1)


cleaned_data = []
processed_count = 0
skipped_count = 0

for i, item in enumerate(data):
    if not isinstance(item, dict):
        logging.warning(f"Skipping item {i+1} as it is not a dictionary.")
        skipped_count += 1
        continue

    logging.info(f"Processing item {i+1}/{len(data)} (URL: {item.get('url', 'N/A')})...")
    cleaned_item = {}

    original_text = item.get('content_text', '')
    url = item.get('url', '')
    original_title = item.get('title', '')
    original_date = item.get('date') # Allows non-null original dates

    # Clean content
    cleaned_text = clean_review_text(original_text, url)
    if not cleaned_text:
        logging.warning(f"Skipping item {i+1} due to empty cleaned text (URL: {url}).")
        skipped_count += 1
        continue

    cleaned_item['content_text'] = cleaned_text
    cleaned_item['url'] = url

    # Extract Title
    title = original_title.strip() if original_title else ''
    if not title:
        title = extract_title_from_url(url)
        # Fallback: Check first line of cleaned text
        if not title and cleaned_text:
             lines = cleaned_text.split('\n', 1)
             first_line = lines[0].strip()
             # Heuristic: Check if first line looks like a plausible title
             if 5 < len(first_line) < 120 and not first_line.startswith('['): # Avoid bracketed placeholders
                 title = first_line
                 logging.info(f"  Extracted title from first line: '{title}'")

    cleaned_item['title'] = title if title else "Unknown Title"

    # Extract Date (only if original is None/null)
    date_val = original_date
    if date_val is None and url:
        date_val = extract_date_from_url(url)
        if date_val:
             logging.info(f"  Extracted date from URL: {date_val}")

    cleaned_item['date'] = date_val # Will be original date, extracted YYYY-MM, or None

    cleaned_data.append(cleaned_item)
    processed_count += 1

logging.info(f"Finished processing. Processed: {processed_count}, Skipped: {skipped_count}.")

# Save cleaned data
try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    logging.info(f"Cleaned data successfully saved to '{output_filename}'")
except Exception as e:
    logging.error(f"Failed to write cleaned data to '{output_filename}': {e}")