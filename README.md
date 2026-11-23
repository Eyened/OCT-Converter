# OCT-Converter (Eyened fork)

Eye scanners store an OCT scan in the manufacturer’s own file format. This package reads those files and writes DICOM, the format hospital image systems use. It accepts Topcon (`.fda`, `.fds`), Heidelberg (`.e2e`), Zeiss (`.img`), and Bioptigen (`.oct`) scans.

Previewing a volume, saving a movie, or dumping the raw metadata is unchanged from the original project. That documentation is in the [upstream README](https://github.com/marksgraham/OCT-Converter#readme). This copy, maintained by Eyened from [marksgraham/OCT-Converter](https://github.com/marksgraham/OCT-Converter), is aimed at producing DICOM that a hospital viewer can file and open.

## What is different here

### Convert a scan from the command line

Give the program a scan file, or a folder of them. It writes the DICOM files for you. You do not need to write a Python script for the common case.

```bash
oct-converter scan.fda -o out/
```

### One examination stays one set of files

A single scan (e.g. an `.fda` file) can hold several images: the OCT, the fundus photo taken with it, a map of the retinal layers, and, for OCTA, a flow volume. Each of those becomes its own DICOM file, and they stay tied to the same examination. From `scan.fda` that is `scan.dcm`, `scan_fundus.dcm`, `scan_seg.dcm`, and, when the scan has flow data, `scan_octa.dcm`.

### Standardized names

The files use the standard names for the scan shape (a line, a cube, a circle, and so on), for retinal layer boundaries, and for which eye was imaged. Distances in the file follow the way that scan was taken. The scanner model is recorded. Its light wavelength is written only when that number is the one the DICOM standard asks for.

### Smaller files, same pixels

The pictures are stored with lossless JPEG 2000 compression, so the file is smaller and the pixels are unchanged. A picture that was already stored that way inside the scanner file is copied through as-is.

### Replace the patient name or id

From Python you can write a different patient name or id into the DICOM. The command line has no option for this; it writes the name from the file. The name is family name, a `^`, then given name.

```python
from oct_converter.dicom import create_dicom_from_oct

create_dicom_from_oct(
    "scan.fda",
    output_dir="out",
    overrides={"PatientName": "Doe^Jane", "PatientID": "P001"},
)
```

## Usage

Python 3.7 or newer, as declared by the package.

```bash
pip install "oct_converter @ git+https://github.com/Eyened/OCT-Converter.git"
```

Convert one file, or every supported file in a directory:

```bash
oct-converter scan.fda -o out/
oct-converter /path/to/scans -o out/ -r
```

A single file is written into `-o`. Several files each get a subdirectory named after the file stem, so companion names such as `*_octa.dcm` do not collide. The command prints each path it writes.

The same entry point from Python:

```python
from oct_converter.dicom import create_dicom_from_oct

paths = create_dicom_from_oct("scan.fda", output_dir="out")
```

`paths` lists every DICOM file written for that scan. For an FDA scan that is the OCT volume (`scan.dcm`) and, when the file contained them, `scan_fundus.dcm`, `scan_seg.dcm`, and `scan_octa.dcm`.

To replace the name stored in the scanner file, pass `overrides` as in the section above. The same call accepts `PatientSex`, `PatientBirthDate`, `Laterality`, `StudyInstanceUID`, and `SeriesInstanceUID`.

## Clinical use

Images and DICOM written with this package are not guaranteed to match the manufacturer’s software. Clinical use is at the user’s own risk.

## License

MIT. Copyright (c) 2021 Mark Graham. See [LICENSE](LICENSE).
