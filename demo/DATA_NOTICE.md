# Demonstration data

Only this selected 28-frame interval is included. The full image collection,
remaining annotations, and all trained checkpoints are private. The demonstration
images and labels are research examples; the source-code MIT license does not
grant a blanket license to the underlying dataset. Contact the corresponding
author for dataset access or redistribution permission.

Images were resized to 1320 by 989, re-encoded as quality-90 JPEG, and stripped
of EXIF, XMP and GPS metadata. Masks use palette-indexed PNG with labels 0-4 and
255 (ignore), resized by nearest neighbor. Dense `gt/` is used only for evaluation;
training and propagation can read only the two masks in `keyframes/`.

`precomputed/` contains saved paper predictions from uniform 5%, run 0. It must
not be mistaken for predictions from the separately executed two-keyframe GPU
demo in `expected_full_demo/`.
