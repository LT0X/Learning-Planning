from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

# 使用 AutoTokenizer 和 AutoModel
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# 准备输入句子
sentence = "apple cat dog"  # 6207, 4937, 3899
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"]  # shape: (1, seq_len)

# 提取 embedding 层
embedding_layer = (
    model.get_input_embeddings()
)  # 等价于 model.bert.embeddings.word_embeddings

# 获取 input embedding（直接查表）
input_embeddings = embedding_layer(input_ids)  # shape: (1, seq_len, 768)

vec_i = input_embeddings[0, 1]  # 'apple'
vec_me = input_embeddings[0, 2]  # 'cat'
vec_you = input_embeddings[0, 3]  # 'dog'

# 查看每个token 对应的 input_ids值（注意，101和102是bert的特殊符号token，代表[cls]和[sep]）
print(f"input_ids: {input_ids}")

# 打印 token 和对应的向量
tokens = tokenizer.tokenize(sentence)
for i, (token, vector) in enumerate(zip(tokens, input_embeddings[0])):
    print(f"Token {i}: {token}")
    print(vector.detach().numpy())  # 转为 numpy 方便查看
    print("=" * 60)

# 计算余弦相似度
sim_i_me = F.cosine_similarity(vec_i, vec_me, dim=0).item()
sim_i_you = F.cosine_similarity(vec_i, vec_you, dim=0).item()
sim_you_me = F.cosine_similarity(vec_you, vec_me, dim=0).item()

# 打印结果
print(f"Cosine similarity between 'apple' and 'cat': {sim_i_me:.4f}")
print(f"Cosine similarity between 'apple' and 'dog': {sim_i_you:.4f}")
print(f"Cosine similarity between 'dog' and 'cat': {sim_you_me:.4f}")
