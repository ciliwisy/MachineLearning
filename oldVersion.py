import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import joblib
from umap import UMAP

# 1. 加载数据
data = pd.read_excel("./data/cleaned_data.xlsx", sheet_name="Sheet1")

# 2. 数据预处理
X = data.drop(columns=["label"])
y = data["label"]

# 目标变量编码
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. 定义 XGBoost 模型参数
model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42, max_depth=6,
                      learning_rate=0.05, n_estimators=300)

# 5. 训练模型
model.fit(X_train, y_train)

# 6. 获取特征重要性
feature_importance = model.feature_importances_
feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

# 获取最重要的10个特征索引
top_features_idx = feature_importance.argsort()[-10:][::-1]
top_features_names = [feature_names[i] for i in top_features_idx]

print("Top 10 important features:", top_features_names)

# 7. 去除前10个最重要的特征
X_reduced = np.delete(X_scaled, top_features_idx, axis=1)

# 8. 重新训练模型
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reduced, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train_r, y_train_r)

# 9. 获取新的特征重要性
new_feature_importance = model.feature_importances_
new_feature_names = [feature_names[i] for i in range(X.shape[1]) if i not in top_features_idx]

# 获取新的最重要的10个特征索引
new_top_features_idx = new_feature_importance.argsort()[-10:][::-1]
new_top_features_names = [new_feature_names[i] for i in new_top_features_idx]

print("New Top 10 important features:", new_top_features_names)

# 10. 可视化新的10个重要特征的影响力
plt.figure(figsize=(10, 6))
plt.barh(range(len(new_top_features_names)), new_feature_importance[new_top_features_idx], align="center")
plt.yticks(range(len(new_top_features_names)), new_top_features_names)
plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance After Removing Top 10")
plt.show()

# 定义颜色和标记
colors = ['#FF6B6B', '#4ECDC4']
markers = ['o', 's']

# 获取模型的预测概率
y_proba = model.predict_proba(X_reduced)

# 计算预测的置信度
prediction_confidence = np.max(y_proba, axis=1)

# 使用模型预测作为标签，而不是原始标签
y_pred = model.predict(X_reduced)

# 计算准确率
accuracy = accuracy_score(y, y_pred)

umap = UMAP(
    n_components=2,
    random_state=42,
    n_neighbors=5,  # 显著减少邻居数，使聚类更分散
    min_dist=0.1,  # 保持适中的最小距离
    metric='euclidean',
    n_epochs=500,  # 增加训练轮数
    learning_rate=0.1,  # 降低学习率以获得更稳定的结果
    spread=0.5,  # 减小spread参数使聚类更紧密
    repulsion_strength=5.0,  # 增加排斥力
    negative_sample_rate=30,  # 增加负采样率
    transform_queue_size=4,  # 增加转换队列大小
    n_jobs=-1,
    verbose=True
)

# 构建增强的特征矩阵
X_combined = np.hstack([
    X_reduced * 0.1,
    y_proba * 5.0,
    prediction_confidence.reshape(-1, 1) * 3.0
])

X_umap = umap.fit_transform(X_combined)

# 改进可视化
plt.figure(figsize=(12, 8))

# 分别绘制高置信度和低置信度的点
confidence_threshold = 0.8

for i, label in enumerate(np.unique(y_pred)):
    mask = y_pred == label
    confidence = prediction_confidence[mask]

    # 绘制高置信度的点
    high_conf_mask = confidence >= confidence_threshold
    if np.any(high_conf_mask):
        plt.scatter(X_umap[mask][high_conf_mask, 0],
                    X_umap[mask][high_conf_mask, 1],
                    c=[colors[i]],
                    marker=markers[i],
                    label=f'Class {label} (High Conf.)',
                    alpha=0.9,
                    s=120,
                    edgecolors='white',
                    linewidth=0.5)

    # 绘制低置信度的点
    low_conf_mask = confidence < confidence_threshold
    if np.any(low_conf_mask):
        plt.scatter(X_umap[mask][low_conf_mask, 0],
                    X_umap[mask][low_conf_mask, 1],
                    c=[colors[i]],
                    marker=markers[i],
                    label=f'Class {label} (Low Conf.)',
                    alpha=0.3,
                    s=80,
                    edgecolors='white',
                    linewidth=0.5)

plt.title(f"UMAP Visualization of Data\nModel Accuracy: {accuracy:.2f}",
          fontsize=16, pad=20)
plt.xlabel("UMAP Component 1", fontsize=14)
plt.ylabel("UMAP Component 2", fontsize=14)

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.2)

# 添加图例
plt.legend(fontsize=12, markerscale=1.5,
           title="Classes", title_fontsize=12)

# 调整布局
plt.tight_layout()

# # 保存高质量图像
# plt.savefig('umap_visualization.png',
#             dpi=300,
#             bbox_inches='tight',
#             facecolor='white',
#             edgecolor='none')

plt.show()

# 改进 t-SNE 可视化
X_combined_tsne = np.hstack([
    X_reduced * 0.1,
    y_proba * 5.0,
    prediction_confidence.reshape(-1, 1) * 3.0
])

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30,
    n_iter=2000,  # 增加迭代次数
    learning_rate=200,  # 调整学习率
    early_exaggeration=12  # 增加早期夸大
)
X_tsne = tsne.fit_transform(X_combined_tsne)

# 使用与 UMAP 相同的可视化风格
plt.figure(figsize=(12, 8))
for i, label in enumerate(np.unique(y_pred)):
    mask = y_pred == label
    confidence = prediction_confidence[mask]

    # 绘制高置信度点
    high_conf_mask = confidence >= confidence_threshold
    if np.any(high_conf_mask):
        plt.scatter(X_tsne[mask][high_conf_mask, 0],
                    X_tsne[mask][high_conf_mask, 1],
                    c=[colors[i]], marker=markers[i],
                    label=f'Class {label} (High Conf.)',
                    alpha=0.9, s=120)

    # 绘制低置信度点
    low_conf_mask = confidence < confidence_threshold
    if np.any(low_conf_mask):
        plt.scatter(X_tsne[mask][low_conf_mask, 0],
                    X_tsne[mask][low_conf_mask, 1],
                    c=[colors[i]], marker=markers[i],
                    label=f'Class {label} (Low Conf.)',
                    alpha=0.3, s=80)

plt.title('t-SNE Visualization with Confidence Information')
plt.grid(True, linestyle='--', alpha=0.2)
plt.legend()
plt.show()