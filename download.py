import datasets
from pathlib import Path
import argparse
import pandas as pd

DATASETS = [
    # source, destination
    (('pauri32/fiqa-2018', None), 'fiqa-2018'),
    (('FinGPT/fingpt-finred', None), 'fingpt-finred'),
    (('zeroshot/twitter-financial-news-sentiment', None), 'twitter-financial-news-sentiment'),
    (('oliverwang15/news_with_gpt_instructions', None), 'news_with_gpt_instructions'),
    (('financial_phrasebank', 'sentences_50agree'), 'financial_phrasebank-sentences_50agree'),
    (('FinGPT/fingpt-fiqa_qa', None), 'fingpt-fiqa_qa'),
    (('FinGPT/fingpt-headline-cls', None), 'fingpt-headline-cls'),
    (('FinGPT/fingpt-finred', None), 'fingpt-finred'),
    (('FinGPT/fingpt-convfinqa', None), 'fingpt-convfinqa'),
    (('FinGPT/fingpt-finred-cls', None), 'fingpt-finred-cls'),
    (('FinGPT/fingpt-ner', None), 'fingpt-ner'),
    (('FinGPT/fingpt-headline', None), 'fingpt-headline-instruct'),
    (('FinGPT/fingpt-finred-re', None), 'fingpt-finred-re'),
    (('FinGPT/fingpt-ner-cls', None), 'fingpt-ner-cls'),
    (('FinGPT/fingpt-fineval', None), 'fingpt-fineval'),
    (('FinGPT/fingpt-sentiment-cls', None), 'fingpt-sentiment-cls'),
]


def download(no_cache: bool = False):
    """Downloads all datasets and saves them as CSV files to where the FinGPT library is located."""
    data_dir = Path(__file__).parent

    for src, dest in DATASETS:
        dest_path = data_dir / dest
        # Kiểm tra nếu thư mục đã tồn tại và không sử dụng no_cache


        # Tải dataset
        dataset = datasets.load_dataset(*src)

        # Tạo thư mục đích nếu chưa tồn tại
        dest_path.mkdir(parents=True, exist_ok=True)

        # Lưu từng split của dataset thành file CSV
        for split_name, split_data in dataset.items():
            # Chuyển dataset thành DataFrame
            df = pd.DataFrame(split_data)
            # Định dạng tên file CSV
            csv_path = dest_path / f"{split_name}.csv"
            # Lưu dưới dạng CSV
            df.to_csv(csv_path, index=False)
            print(f"Saved {split_name} split to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_cache", action="store_true", help="Redownloads all datasets if set to True")

    args = parser.parse_args()
    download(no_cache=args.no_cache)