import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_spatial_heatmap(file_path, gene_name):
    try:
        df = pd.read_csv(file_path, index_col=0)

        if gene_name not in df.columns:
            print(f"错误：基因 '{gene_name}' 不在数据文件中。")
            return

        raw_data = df[gene_name]
        log_transformed_data = np.log1p(raw_data)

        max_val = log_transformed_data.max()
        min_val = log_transformed_data.min()

        if max_val == min_val:
            normalized_val = log_transformed_data * 0
        else:
            normalized_val = (-min_val + log_transformed_data) / (max_val - min_val)

        coords = df.index.to_series().str.split('x', expand=True)
        coords.columns = ['S1', 'S2']
        coords = coords.astype(float)

        plot_df = coords.copy()
        plot_df['value'] = normalized_val

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(plot_df['S1'], plot_df['S2'],
                              c=plot_df['value'],
                              cmap='Spectral_r',
                              s=40,
                              marker='s')

        cbar = plt.colorbar(scatter)
        cbar.set_label('Normalized Expression')

        plt.title(f'Spatial Heatmap: {gene_name}', fontsize=15)
        plt.xlabel('S1', fontsize=12)
        plt.ylabel('S2', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"运行出错: {e}")


csv_file = 'PDAC_A_ST1_processed_data_mincell5.csv'

target_gene = input("请输入想要查询的基因名称: ").strip()

plot_spatial_heatmap(csv_file, target_gene)