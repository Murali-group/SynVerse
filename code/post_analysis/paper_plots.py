import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text

def plot_model_comparison_with_deepsynergy(filename):
    df = pd.read_csv(filename, sep='\t')[['Model name', 'Own', 'DeepSynergy']]
    # Create the scatter plot

    plt.figure(figsize=(8, 6))
    plt.scatter(df['Own'], df['DeepSynergy'], cmap='viridis')
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    # Plot the diagonal line (y=x) to compare the performances
    plt.plot([0, 1], [0,1], 'r--', label='Equal Performance Line')

    # Annotate each point with the model name
    texts = []
    for i, row in df.iterrows():
        text = plt.text(row['Own'] + 0.01, row['DeepSynergy'], row['Model name'], fontsize=9)
        texts.append(text)

    # Adjust the text positions to avoid overlap using adjustText
    adjust_text(texts)
    # Add labels and title
    plt.xlabel('Model Own Performance')
    plt.ylabel('DeepSynergy Performance')
    plt.title('Comparison of Model and DeepSynergy Performance')

    # Add a legend
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()




plot_model_comparison_with_deepsynergy("/home/grads/tasnina/Projects/SynVerse/inputs/existing_model_performance.tsv")

