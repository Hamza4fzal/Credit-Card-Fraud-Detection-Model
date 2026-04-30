import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Loading raw data for visualization...")
    # Load dataset
    df = pd.read_csv('creditcard.csv')
    
    print("Generating class distribution plot...")
    # Set the style
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    
    # Create a countplot for the 'Class' column
    ax = sns.countplot(x='Class', data=df, palette='Set2')
    
    # Add title and labels
    plt.title('Class Distribution (0: Valid, 1: Fraud)', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    # Add counts on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')
    
    # Save the plot
    output_filename = 'class_distribution.png'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Visualization saved successfully as '{output_filename}'.")

if __name__ == "__main__":
    main()
