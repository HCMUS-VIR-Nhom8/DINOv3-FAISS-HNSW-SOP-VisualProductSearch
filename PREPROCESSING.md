# Các bước tiền xử lý

SOP 120k img<br>
    => 1. Stratified sample 20k image<br>
    => 2. Parse metadata (image_id,class_id super_class_id path)<br>
    => 3. Image resize aspect ratio<br>
    => 4.blur detection / JPEG artifact<br>
    => 5. Object localization (segmentation/detection)<br>
    => 6. Crop + padding object normalization<br>
    => 7. Geometric norm / EXIF / orientation / letterbox<br> 
    => 8. Illumination correction<br> 
=> 20k processed image<br>
=> DINOv3 feature extraction => …
