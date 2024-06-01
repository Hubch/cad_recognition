import fitz

def pdf_to_image(pdf_path, output_path):
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        image_list = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        
        image_path = output_path.format(page_num)
        image_list.save(image_path)
    
    pdf_document.close()



if __name__ == "__main__":
    
    # 指定PDF文件路径和输出图片的文件名格式
    pdf_path = 'I:/company project/cad_recognition/docs/02A.pdf'
    output_path = 'I:/company project/cad_recognition/backend/run_logs/output_page.png'
    pdf_to_image(pdf_path, output_path)