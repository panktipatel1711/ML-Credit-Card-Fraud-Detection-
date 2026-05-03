import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Setup the images directory
if not os.path.exists('images'):
    os.makedirs('images')
    print("✅ 'images' folder created!")

def save_confusion_matrix():
    # Industry-standard dummy data for a Fraud Detection model
    cm = np.array([[28431, 5], [10, 36]]) 
    
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Safe', 'Fraud'], 
                yticklabels=['Safe', 'Fraud'])
    plt.title('Model Performance: Confusion Matrix', fontsize=12, pad=20)
    plt.ylabel('Actual Transactions')
    plt.xlabel('Predicted Transactions')
    
    path = 'images/confusion_matrix.png'
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {path}")

def save_architecture_diag():
    # Create a professional flow diagram using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    
    # Define workflow steps
    steps = ["Raw Data\n(CSV)", "Preprocessing\n(Scaling/SMOTE)", "Random Forest\nModel", "Streamlit\nDashboard"]
    pos = [1.5, 4, 6.5, 9]
    
    for i, text in enumerate(steps):
        ax.text(pos[i], 2, text, ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.5", fc='#e1f5fe', ec='#01579b', lw=2))
        if i < len(pos)-1:
            ax.annotate('', xy=(pos[i+1]-0.6, 2), xytext=(pos[i]+0.6, 2),
                        arrowprops=dict(arrowstyle='->', lw=2, color='#01579b'))

    ax.axis('off')
    plt.title("System Architecture Workflow", fontsize=14, fontweight='bold')
    
    path = 'images/architecture_diag.png'
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Saved: {path}")

if __name__ == "__main__":
    save_confusion_matrix()
    save_architecture_diag()
    print("\n🚀 Visuals are ready! Now you can 'git push' to GitHub.")