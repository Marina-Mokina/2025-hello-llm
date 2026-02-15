"""
Fine-tuning starter.
"""

# pylint: disable=too-many-locals, undefined-variable, unused-import, too-many-branches, too-many-statements
import json
from pathlib import Path

from core_utils.llm.time_decorator import report_time
from lab_8_sft.main import LLMPipeline, RawDataImporter, RawDataPreprocessor, TaskDataset

BASE_PATH = Path(__file__).parent


@report_time
def main() -> None:
    """
    Run the translation pipeline.
    """
    with open(BASE_PATH / 'settings.json', 'r', encoding='utf-8') as f:
        settings = json.load(f)

    importer = RawDataImporter(settings['parameters']['dataset'])
    importer.obtain()
    if importer.raw_data is None:
        return

    preprocessor = RawDataPreprocessor(importer.raw_data)

    result = preprocessor.analyze()
    print('Dataset analysis:')
    for key, value in result.items():
        print(f'{key}: {value}')

    preprocessor.transform()
    dataset = TaskDataset(preprocessor.data.head(100))

    pipeline = LLMPipeline(settings['parameters']['model'], dataset, 120, 1, 'cpu')

    print('\nModel analysis:')
    for key, value in pipeline.analyze_model().items():
        print(f'{key}: {value}')

    assert dataset is not None, "Demo does not work correctly"


if __name__ == "__main__":
    main()
