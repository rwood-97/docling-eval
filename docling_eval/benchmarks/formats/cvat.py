from pathlib import Path

def from_cvat_to_docling_document(annotation_filenames:List[Path], page_images:Path):
    pass
    

def from_docling_document_to_cvat(docs: List[DoclingDocument],
                                  output_dir:Path,
                                  labels: Set[DocItemLabel] = DEFAULT_EXPORT_LABELS):    

    output_page_dir = output_dir / "page_images"

    for _ in [output_dir, output_page_dir]:
        os.makedirs(_, exist_ok=True)

    img_id = 0
    for doc_id,doc in enumerate(docs):

        page_images = []
        page_fnames = []
        for j,page in enumerate(doc.pages):
            page_image = page.pil_image
            page_image.save(str(output_page_dir / f"doc_{i:06}_{j:06}.png"))
            
            page_images.append(page_image)
            page_fnames.append(f"doc_{i:06}_{j:06}.png")

        page_bboxes = {i: [] for i,fname in enumerate(page_fnames)}
        for item, level in doc.iterate_items():
            if isinstance(item, DocItem) and item.label in labels:
                for prov in item.prov:
                    page_no = prov.page_no

                    page_w = doc.pages[prov.page_no].size.width
                    page_i = doc.pages[prov.page_no].size.height

                    img_w = page_images[page_no-1].width
                    img_h = page_images[page_no-1].height
                    
                    page_bbox = prov.bbox 
                    origin = bbox.coord_origin
                    
                    img_bbox = [
                        page_bbox.l/page_w*img_w,
                        page_bbox.b/page_h*img_h,
                        page_bbox.r/page_w*img_w,
                        page_bbox.t/page_h*img_h,
                    ]

                    page_bboxes[page_no-1].append({
                        "label": item.label,
                        "bbox": img_bbox,
                    })

        
                    
        results=[]
        for page_no, page_file in enumerate(page_fnames):
            img_w = page_images[page_no].width
            img_h = page_images[page_no].height
            
            results.append(f'<image id="{img_id}" name="{page_file}" width="{img_w}" height="{img_h}">')

            for bbox_id, page_bbox in enumerate(page_bboxes):
                results.append(f'<bbox label="{label}" source="manual" occluded="0" xtl="{}" ytl="{}" xbr="{}" ybr="{}" z_order="{bbox_id}">')
            
            results.append("</image>")
