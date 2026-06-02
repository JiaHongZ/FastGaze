import numpy as np
import torch
import CLIPmain.clip as clip
from PIL import Image
import os
def get_clip_embedding(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text = clip.tokenize(text).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        print(text_features.shape)
    return text_features

embedding_dict = np.load(open(os.path.join('/zjh/data/Gazeformer-CVPR-2023/dataset', 'embeddings.npy'), mode='rb'), allow_pickle = True).item()
print(embedding_dict.keys())
print(embedding_dict['potted plant'].shape) # (768,)

text_features = get_clip_embedding(embedding_dict.keys())
for i,(key,value) in enumerate(embedding_dict.items()):
    embedding_dict[key] = text_features[i].cpu().numpy().astype(np.float32)

# 打印更新后的嵌入形状
print(embedding_dict.keys())
print(embedding_dict['potted plant'].shape)  # torch.Size([512])

# 保存更新后的嵌入字典到 .npy 文件
np.save(os.path.join('/zjh/data/Gazeformer-CVPR-2023/dataset', 'clip_embeddings.npy'), embedding_dict)





# image = preprocess(Image.open("F:\project\Gazeformer-main\CLIPmain\CLIP.png")).unsqueeze(0).to(device)
# text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
#
# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     print(text_features.shape)
#     logits_per_image, logits_per_text = model(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()
#
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]