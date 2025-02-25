import collections


def auc(scores, labels):
    data = list(zip(scores, labels))
    data.sort(key=lambda x: x[0])  # 按照预测分数从小到大排序

    pos_numbers = sum(labels)  # 正样本数量
    neg_numbers = len(labels) - pos_numbers  # 负样本数量

    auc_value = 0
    sum_neg = 0  # 累计的负样本的数量
    for score, label in data:
        if label == 0:
            sum_neg += 1
        else:
            auc_value += sum_neg  # 正样本得分大于当前负样本的次数

    # 如果正负样本数有一个为0，AUC为0，避免分母为0的情况出现
    if pos_numbers * neg_numbers == 0:
        return 0

    auc_value /= (pos_numbers * neg_numbers)
    return auc_value


def g_auc(scores, labels, user_id):
    data = list(zip(scores, labels, user_id))
    user_datas = collections.defaultdict(list)

    for score, label, user_id in data:
        user_datas[user_id].append((score, label))

    #  计算每个用户的auc
    user_auc = {}
    for user_id, user_data in user_datas.items():
        score, label = zip(*user_data)
        user_auc[user_id] = auc(score, label)

    # weight的值为每个用户的数据的长度，代表每个用户在整体g_auc中的贡献
    weight = sum(len(data) for data in user_datas.values())
    g_auc_value = sum(len(data) * auc_value for data, auc_value in zip(user_datas.values(), user_auc.values())) / weight
    return g_auc_value

# scores_value = [0.9, 0.8, 0.7, 0.6, 0.5]
# labels_value = [0, 1, 0, 1, 1]
# user_ids = [2, 2, 3, 2, 3]
# print(auc(scores_value, labels_value))
# print(g_auc(scores_value, labels_value, user_ids))
