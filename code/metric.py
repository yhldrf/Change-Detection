from __future__ import print_function
import numpy as np
from pprint import pprint
from sklearn.metrics import confusion_matrix as skl_get_confusion_matrix
import matplotlib.pyplot as plt
import itertools


class ConfusionMatrix:
    def __init__(self, num_classes):
        """
        Standard 0-based label system:
        label = {0, 1, 2, ..., num_classes-1}
        """
        self.num_classes = num_classes
        self.confusion_matrix = np.zeros(
            (self.num_classes, self.num_classes), dtype=np.int64
        )
        self.valid_labels = set(range(self.num_classes))

    # =========================
    # 修改1：单样本累加（保持）
    # =========================
    def increment(self, gt_label, pd_label):
        if gt_label not in self.valid_labels:
            raise ValueError("Invalid gt_label")
        if pd_label not in self.valid_labels:
            raise ValueError("Invalid pd_label")
        self.confusion_matrix[gt_label][pd_label] += 1

    # =========================
    # 修改2：批量累加（保持sklearn）
    # =========================
    def increment_from_list(self, gt_labels, pd_labels):
        increment_cm = skl_get_confusion_matrix(
            gt_labels, pd_labels, labels=list(range(self.num_classes))
        )
        self.confusion_matrix += increment_cm

    # =========================
    # 修改3：IoU（去掉1-based逻辑）
    # =========================
    def get_per_class_ious(self):
        ious = []
        for c in range(self.num_classes):
            tp = self.confusion_matrix[c, c]
            fp = np.sum(self.confusion_matrix[:, c]) - tp
            fn = np.sum(self.confusion_matrix[c, :]) - tp

            union = tp + fp + fn
            if union == 0:
                ious.append(0.0)
            else:
                ious.append(tp / union)
        return ious

    # =========================
    # 修改4：mean IoU
    # =========================
    def get_mean_iou(self):
        return np.mean(self.get_per_class_ious())

    # =========================
    # 修改5：overall accuracy
    # =========================
    def get_accuracy(self):
        return np.trace(self.confusion_matrix) / np.sum(self.confusion_matrix)

    # =========================
    # 修改6：per-class accuracy（关键修复NaN）
    # =========================
    def get_per_class_accuracy(self):
        denom = np.sum(self.confusion_matrix, axis=1)
        denom = np.where(denom == 0, 1, denom)  # 防 NaN
        return np.diag(self.confusion_matrix) / denom

    # =========================
    # 修改7：打印函数
    # =========================
    def print_metrics(self, labels=None):
        print("Confusion matrix:")

        if labels is None:
            labels = [str(i) for i in range(self.num_classes)]

        width = max([len(x) for x in labels] + [7])

        print(" " * (width + 4), end="")
        for l in labels:
            print(f"{l:>{width}}", end=" ")
        print()

        for i, l in enumerate(labels):
            print(f"{l:>{width}}", end=" ")
            for j in range(self.num_classes):
                print(f"{self.confusion_matrix[i, j]:>{width}.0f}", end=" ")
            print()

        print("\nIoU per class:")
        pprint(self.get_per_class_ious())

        print("\nmIoU:")
        print(self.get_mean_iou())

        print("\nAccuracy:")
        print(self.get_accuracy())

        print("\nPer-class accuracy:")
        print(self.get_per_class_accuracy())


# =========================
# 可视化函数（修复bug）
# =========================
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i,
            format(cm[i, j], 'd'),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black"
        )

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()