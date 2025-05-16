import streamlit as st
import pandas as pd
import tempfile
import os
import cv2
import numpy as np
import fitz  # PyMuPDF
import pyzbar.pyzbar as pyzbar
import io
from PIL import Image
import re

def preprocess_image(image):
    """Preprocess image to improve barcode detection"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Optional: noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return opening

def extract_barcodes_from_image(image):
    """Extract barcodes from an image"""
    # Try with original image
    barcodes = pyzbar.decode(image)
    
    if not barcodes:
        # If no barcodes found, try with preprocessed image
        processed_img = preprocess_image(image)
        barcodes = pyzbar.decode(processed_img)
    
    results = []
    for barcode in barcodes:
        barcode_data = barcode.data.decode('utf-8')
        # Clean the barcode data - remove non-alphanumeric characters
        barcode_data = re.sub(r'[^A-Za-z0-9]', '', barcode_data)
        results.append(barcode_data)
    
    return results

def extract_barcodes_from_pdf(pdf_file):
    """Extract barcodes from all pages of a PDF file"""
    all_barcodes = []
    
    # Open the PDF
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    
    # Process each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        
        # Get page as image
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img_bytes = pix.tobytes()
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_np is None:
            # Try another approach if the first one fails
            img = Image.open(io.BytesIO(img_bytes))
            img_np = np.array(img)
        
        # Extract barcodes
        barcodes = extract_barcodes_from_image(img_np)
        all_barcodes.extend(barcodes)
    
    pdf_document.close()
    return all_barcodes

def clean_barcode(barcode):
    """Clean and validate barcode format"""
    # Remove prefix URLs and any unwanted characters
    cleaned = re.sub(r'httpsapiflashexpresscomwebproofgo', '', barcode, flags=re.IGNORECASE)
    cleaned = re.sub(r'[^A-Za-z0-9]', '', cleaned)
    
    # Check if the barcode follows expected patterns (AYUxxxxxxx or PHTxxxxxxx)
    cleaned_upper = cleaned.upper()
    if re.match(r'^(AYU|PHT|RAT)[A-Z0-9]{7,}$', cleaned_upper):
        return cleaned_upper

    return None  # Invalid barcode

def main():
    st.title("PDF Barcode Extractor")
    st.write("อัปโหลดไฟล์ PDF ที่มีบาร์โค้ดเพื่อดึงข้อมูลและบันทึกเป็น Excel")
    
    uploaded_files = st.file_uploader("เลือกไฟล์ PDF", type=['pdf'], accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("ประมวลผล"):
            all_barcodes = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"กำลังประมวลผลไฟล์ {file.name}...")
                
                try:
                    barcodes = extract_barcodes_from_pdf(file)
                    # Clean and validate barcodes
                    valid_barcodes = [bc for bc in barcodes if bc]
                    all_barcodes.extend(valid_barcodes)
                    
                    st.write(f"พบบาร์โค้ด {len(valid_barcodes)} รายการจากไฟล์ {file.name}")
                except Exception as e:
                    st.error(f"เกิดข้อผิดพลาดในการประมวลผลไฟล์ {file.name}: {str(e)}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("การประมวลผลเสร็จสิ้น")
            
            if all_barcodes:
                # Remove duplicates while preserving order
                unique_barcodes = []
                for bc in all_barcodes:
                    if bc not in unique_barcodes:
                        unique_barcodes.append(bc)
                
                # Clean barcodes further
                cleaned_barcodes = [bc for bc in unique_barcodes if clean_barcode(bc)]
                
                # Create DataFrame without headers
                df = pd.DataFrame(cleaned_barcodes)
                
                # Display results
                st.write(f"สรุป: พบบาร์โค้ดทั้งหมด {len(cleaned_barcodes)} รายการ")
                st.dataframe(df)
                
                # Create CSV file
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, header=False)
                csv_buffer.seek(0)
                
                # Provide download button
                st.download_button(
                    label="ดาวน์โหลด CSV",
                    data=csv_buffer.getvalue().encode('utf-8'),
                    file_name="output.csv",
                    mime="text/csv"
                )
            else:
                st.warning("ไม่พบบาร์โค้ดในไฟล์ที่อัปโหลด")

if __name__ == "__main__":
    main()
