"""
Starter for demonstration of laboratory work.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import
import json
from lab_7_llm.main import report_time, RawDataImporter, RawDataPreprocessor


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open('settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    dataset_name = settings["parameters"]["dataset"]

    importer = RawDataImporter(dataset_name)
    importer.obtain()

    preprocessor = RawDataPreprocessor(importer.raw_data)

    result = preprocessor.analyze()

    for key, value in result.items():
        print(f"{key}: {value}")

    assert result is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
