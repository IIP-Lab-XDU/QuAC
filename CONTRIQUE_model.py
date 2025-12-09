import torch.nn as nn
import torch
import torchvision

class CONTRIQUE_model(nn.Module):
    # resnet50 architecture with projector
    def __init__(self, n_features, \
                 patch_dim = (2,2), normalize = True, projection_dim = 128):
        super(CONTRIQUE_model, self).__init__()

        self.normalize = normalize
        # self.encoder = nn.Sequential(*list(encoder.children())[:-2])
        self.encoder = nn.Sequential(*list(torchvision.models.resnet50(pretrained=False).children())[:-2])
        self.n_features = n_features
        self.patch_dim = patch_dim
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool_patch = nn.AdaptiveAvgPool2d(patch_dim)

        # MLP for projector
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.BatchNorm1d(self.n_features),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
            nn.BatchNorm1d(projection_dim),
        )
        
    def forward(self, x_i, x_j):
        # global features
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        # local features
        h_i_patch = self.avgpool_patch(h_i)
        h_j_patch = self.avgpool_patch(h_j)
        
        h_i_patch = h_i_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])
        
        h_j_patch = h_j_patch.reshape(-1,self.n_features,\
                                      self.patch_dim[0]*self.patch_dim[1])
        
        h_i_patch = torch.transpose(h_i_patch,2,1)
        h_i_patch = h_i_patch.reshape(-1, self.n_features)
        
        h_j_patch = torch.transpose(h_j_patch,2,1)
        h_j_patch = h_j_patch.reshape(-1, self.n_features)
        
        h_i = self.avgpool(h_i)
        h_j = self.avgpool(h_j)

        q = h_i
        
        h_i = h_i.view(-1, self.n_features)
        h_j = h_j.view(-1, self.n_features)
        
        if self.normalize:
            h_i = nn.functional.normalize(h_i, dim=1)
            h_j = nn.functional.normalize(h_j, dim=1)
            
            h_i_patch = nn.functional.normalize(h_i_patch, dim=1)
            h_j_patch = nn.functional.normalize(h_j_patch, dim=1)
        
        # global projections
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        # local projections
        z_i_patch = self.projector(h_i_patch)
        z_j_patch = self.projector(h_j_patch)
        
        return z_i, z_j, z_i_patch, z_j_patch, h_i, h_j, h_i_patch, h_j_patch, q
        
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机生成示例输入
    inputs = torch.randn(8, 3, 512, 512)
    inputs_2 = torch.randn(8, 3, 512, 512)

    # 模型权重路径
    model_path = "/path/to/CONTRIQUE_checkpoint25.tar"
    # download from https://github.com/pavancm/CONTRIQUE

    # 初始化 CONTRIQUE 模型
    quality_model = CONTRIQUE_model(n_features=2048)
    quality_model.load_state_dict(torch.load(model_path, map_location=device))
    quality_model.to(device)
    quality_model.eval()  # 不训练

    # 提取特征
    with torch.no_grad():
        _, _, _, _, model_feat, model_feat_2, _, _, q = quality_model(inputs.to(device), inputs_2.to(device))

    print("CONTRIQUE feature shape:", q.shape)  # 输出: torch.Size([8, 2048, 1, 1])
