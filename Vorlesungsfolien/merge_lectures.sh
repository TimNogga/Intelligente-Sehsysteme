#!/bin/bash

OUTPUT_FILE="${1:-merged_lectures.pdf}"
TEMP_DIR="./temp_sort"
MERGE_LIST="$TEMP_DIR/merge_list.txt"

mkdir -p "$TEMP_DIR"

echo "Searching for PDF files with 'Vorlesung' pattern..."

declare -a pdf_files
while IFS= read -r file; do
    if [[ "$file" =~ Vorlesung[[:space:]]*([0-9]+) ]]; then
        lecture_num="${BASH_REMATCH[1]}"
        printf -v padded_num "%03d" "$lecture_num"
        safe_name="${padded_num}_$(basename "$file" | sed 's/ /_/g')"
        new_name="$TEMP_DIR/$safe_name"
        cp "$file" "$new_name"
        pdf_files+=("$new_name")
        echo "Found: $file -> Lecture $lecture_num"
    fi
done < <(find . -maxdepth 1 -name "*.pdf" -type f | sort)

if [ ${#pdf_files[@]} -eq 0 ]; then
    echo "Error: No PDF files with 'Vorlesung X' pattern found!"
    exit 1
fi

printf "%s\n" "${pdf_files[@]}" | sort > "$MERGE_LIST"

echo ""
echo "Merging ${#pdf_files[@]} lecture files in order:"

while IFS= read -r file; do
    original_name=$(basename "$file" | cut -d'_' -f2- | sed 's/_/ /g')
    echo "  - $original_name"
done < "$MERGE_LIST"

echo ""

if command -v pdfunite >/dev/null 2>&1; then
    echo "Merging PDFs using pdfunite..."
    pdfunite $(cat "$MERGE_LIST") "$OUTPUT_FILE"
else
    echo "Error: pdfunite not found. Please install:"
    echo "  Ubuntu/Debian: sudo apt-get install poppler-utils"
    echo "  macOS: brew install poppler"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Successfully created: $OUTPUT_FILE"
    ls -lh "$OUTPUT_FILE"
else
    echo "❌ Error during PDF merge!"
fi

rm -rf "$TEMP_DIR"
