import pandas as pd


def get_significant_genes(file_path, target_cell_type, threshold=0.01):
    try:
        df = pd.read_csv(file_path, index_col=0)

        if target_cell_type not in df.columns:
            print(f"错误：在该文件中找不到名为 '{target_cell_type}' 的细胞类型。")
            print("可选的细胞类型有：", list(df.columns))
            return None

        significant_mask = df[target_cell_type] < threshold

        significant_genes = df.index[significant_mask].tolist()

        return significant_genes

    except FileNotFoundError:
        print("错误：未找到CSV文件，请检查文件路径是否正确。")
    except Exception as e:
        print(f"发生错误：{e}")


file_name = 'Q_save_sub_cancer.csv'

cell_input = input("请输入细胞名称: ").strip()

genes = get_significant_genes(file_name, cell_input)

if genes is not None:
    print(f"\n在细胞类型 '{cell_input}' 中，表达显著的基因（q_value < 0.01）共有 {len(genes)} 个：")
    if len(genes) > 0:
        for gene in genes:
            print(gene)
    else:
        print("未发现显著表达的基因。")