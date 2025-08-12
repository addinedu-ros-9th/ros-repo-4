import matplotlib.pyplot as plt
import numpy as np

# Updated Data - Reasoning Mode (1.2B category)
benchmarks = ["MMLU-Redux", "MMLU-Pro", "GPQA-Diamond", "AIME 2025", "IFEval", "KMMLU-Pro"]
exaone_4_1_2b = [71.5, 59.3, 52.0, 45.2, 67.8, 42.7]
qwen3_1_7b = [73.9, 57.7, 40.1, 36.8, 72.5, 38.3]
smollm3_3b = [74.8, 57.8, 41.7, 36.7, 71.2, 30.5]

# Model information
labels = benchmarks
models = {
    "EXAONE 4.0 1.2B": exaone_4_1_2b,
    "Qwen 3 1.7B": qwen3_1_7b,
    "SmolLM 3 3B": smollm3_3b
}

# Custom colors
colors = ["#009999", "#ff9c1d", "#E84D4D"]

# ------- Radar Chart -------
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(16, 8))

# Radar Chart
ax1 = plt.subplot(131, polar=True)
for i, (model, scores) in enumerate(models.items()):
    data = scores + scores[:1]
    ax1.plot(angles, data, label=model, linewidth=3, color=colors[i])
    ax1.fill(angles, data, alpha=0.15, color=colors[i])

ax1.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=10)
ax1.set_title("Reasoning Mode - Key Benchmarks", fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax1.set_ylim(0, 80)
ax1.grid(True)

# ------- Category Average Bar Chart -------
categories = {
    "World Knowledge": ["MMLU-Redux", "MMLU-Pro", "GPQA-Diamond"],
    "Math & Coding": ["AIME 2025"],
    "Instruction Following": ["IFEval"],
    "Multilinguality": ["KMMLU-Pro"]
}

category_scores = {model: [] for model in models}
for cat, bench_list in categories.items():
    for model, scores in models.items():
        avg = np.mean([scores[labels.index(b)] for b in bench_list])
        category_scores[model].append(avg)

x = np.arange(len(categories))
width = 0.25

ax2 = plt.subplot(132)
for i, (model, scores) in enumerate(category_scores.items()):
    bars = ax2.bar(x + i * width, scores, width, label=model, color=colors[i], alpha=0.8)
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

ax2.set_xticks(x + width)
ax2.set_xticklabels(categories.keys(), rotation=15, ha='right')
ax2.set_ylim(0, 80)
ax2.set_ylabel("Score (%)", fontsize=12)
ax2.set_title("Performance by Category", fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ------- Model Specifications -------
ax3 = plt.subplot(133)
ax3.axis('off')

model_specs = {
    "EXAONE 4.0 1.2B": {"size": "1.28B", "reasoning": "✅", "overall": 56.5},
    "Qwen 3 1.7B": {"size": "1.72B", "reasoning": "✅", "overall": 53.2},
    "SmolLM 3 3B": {"size": "3.08B", "reasoning": "✅", "overall": 52.0}
}

# Create table
table_data = []
headers = ["Model", "Size", "Hybrid Reasoning", "Avg Score"]
for i, (model, specs) in enumerate(model_specs.items()):
    table_data.append([
        model,
        specs["size"],
        specs["reasoning"],
        f"{specs['overall']:.1f}%"
    ])

# Create table
table = ax3.table(cellText=table_data,
                 colLabels=headers,
                 cellLoc='center',
                 loc='center',
                 colColours=['lightgray']*4)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color code the rows
for i in range(len(model_specs)):
    for j in range(len(headers)):
        table[(i+1, j)].set_facecolor(colors[i])
        table[(i+1, j)].set_alpha(0.3)

ax3.set_title("Model Specifications", fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.suptitle("Small Language Models Comparison - Reasoning Mode", fontsize=16, fontweight='bold')
plt.show()

# ------- Additional Detailed Comparison -------
fig2, ((ax4, ax5), (ax6, ax7)) = plt.subplots(2, 2, figsize=(16, 12))

# All benchmarks comparison
all_benchmarks = ["MMLU-Redux", "MMLU-Pro", "GPQA-Diamond", "AIME 2025", "IFEval", "KMMLU-Pro"]
all_scores = {
    "EXAONE 4.0 1.2B": [71.5, 59.3, 52.0, 45.2, 67.8, 42.7],
    "Qwen 3 1.7B": [73.9, 57.7, 40.1, 36.8, 72.5, 38.3],
    "SmolLM 3 3B": [74.8, 57.8, 41.7, 36.7, 71.2, 30.5]
}

x_pos = np.arange(len(all_benchmarks))
width = 0.25

# Detailed benchmark comparison
for i, (model, scores) in enumerate(all_scores.items()):
    bars = ax4.bar(x_pos + i * width, scores, width, label=model, color=colors[i], alpha=0.8)

ax4.set_xlabel('Benchmarks', fontsize=12)
ax4.set_ylabel('Score (%)', fontsize=12)
ax4.set_title('Detailed Benchmark Comparison', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos + width)
ax4.set_xticklabels(all_benchmarks, rotation=45, ha='right')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Performance vs Model Size
model_sizes = [1.28, 1.72, 3.08]  # in billions
avg_scores = [56.5, 53.2, 52.0]
model_names = ["EXAONE 4.0\n1.2B", "Qwen 3\n1.7B", "SmolLM 3\n3B"]

scatter = ax5.scatter(model_sizes, avg_scores, c=colors, s=200, alpha=0.7)
for i, name in enumerate(model_names):
    ax5.annotate(name, (model_sizes[i], avg_scores[i]), 
                xytext=(10, 10), textcoords='offset points', fontsize=10)

ax5.set_xlabel('Model Size (Billions of Parameters)', fontsize=12)
ax5.set_ylabel('Average Score (%)', fontsize=12)
ax5.set_title('Performance vs Model Size', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Strengths comparison (top 3 categories for each model)
strengths = {
    "EXAONE 4.0 1.2B": ["IFEval (67.8)", "MMLU-Redux (71.5)", "MMLU-Pro (59.3)"],
    "Qwen 3 1.7B": ["MMLU-Redux (73.9)", "IFEval (72.5)", "MMLU-Pro (57.7)"],
    "SmolLM 3 3B": ["MMLU-Redux (74.8)", "IFEval (71.2)", "MMLU-Pro (57.8)"]
}

ax6.axis('off')
y_pos = 0.8
for i, (model, strength_list) in enumerate(strengths.items()):
    ax6.text(0.1, y_pos, model, fontsize=14, fontweight='bold', color=colors[i])
    y_pos -= 0.1
    for strength in strength_list:
        ax6.text(0.15, y_pos, f"• {strength}", fontsize=11)
        y_pos -= 0.08
    y_pos -= 0.05

ax6.set_title('Model Strengths (Top 3 Benchmarks)', fontsize=14, fontweight='bold')

# Reasoning vs Non-Reasoning comparison (if we had non-reasoning data)
ax7.axis('off')
ax7.text(0.5, 0.5, 'Reasoning Mode Analysis\n\nAll models show:\n• Strong performance in World Knowledge\n• Competitive Instruction Following\n• Math/Coding challenges remain\n• Multilingual capabilities vary', 
         ha='center', va='center', fontsize=12, 
         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
ax7.set_title('Key Insights', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.suptitle("Comprehensive Small Language Models Analysis", fontsize=16, fontweight='bold')
plt.show()
