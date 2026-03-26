from pathlib import Path
from errant.converter import convert_m2_file


input_files = [
    "original_datasets/bea_2019_dev/ABC.train.gold.bea19.m2",
    "original_datasets/bea_2019_dev/ABCN.dev.gold.bea19.m2",
]

def main():
    orig_suffix = ".orig.txt"
    cor_suffix = ".cor.txt"
    convert_m2_to_txt(orig_suffix=orig_suffix, cor_suffix=cor_suffix)


def convert_m2_to_txt(orig_suffix=".orig.txt", cor_suffix=".cor.txt"):
    for input_file in input_files:
        ip = Path(input_file)
        output_path = Path("testing_datasets")
        output_path.mkdir(parents=True, exist_ok=True)
        result = convert_m2_file(input_file)

        if orig_suffix:
            output_file_orig = output_path / ip.name.replace(".m2", orig_suffix)
            with open(output_file_orig, "w", encoding="utf-8") as f:
                for item in result:
                    f.write(item["original"] + "\n")
            print(f"Saved {output_file_orig}")
            
        if cor_suffix:
            output_file_cor = output_path / ip.name.replace(".m2", cor_suffix)
            with open(output_file_cor, "w", encoding="utf-8") as f:
                for item in result:
                    f.write(item["corrected"] + "\n")
            print(f"Saved {output_file_cor}")


if __name__ == "__main__":
    main()
