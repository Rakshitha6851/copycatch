# TODO: Implement Image Validation for Uploads

## Steps to Complete

1. **Add is_valid_image function to app.py** ✅
   - Implement function to detect human faces, cat faces using OpenCV Haar cascades.
   - Use pytesseract to check for text presence.
   - Return False if faces detected or no text found (indicating photographic images).

2. **Modify /upload route in app.py** ✅
   - After saving each file, validate if it's an image.
   - Collect invalid image filenames.
   - If invalid images found, delete them from uploads folder and return error message "invalid" in upload.html.
   - Proceed with processing only if all images are valid.

3. **Test the implementation** ✅
   - Run the app and test with various image types (text-based, photos of humans/cats/objects).
   - Verify error message appears for invalid images.
   - Adjust detection parameters if needed for better accuracy.

## Notes
- Dog detection not implemented due to lack of standard Haar cascade; consider adding a custom model if required.
- Non-living things are invalidated if no text is detected (assuming photos without text).
