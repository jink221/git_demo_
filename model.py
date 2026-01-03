import torch
import torch.nn as nn
import torchvision.models as models

# 1. 통합 Encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size, encoder_type="resnet50"):
        super(EncoderCNN, self).__init__()
        self.encoder_type = encoder_type.lower()
        
        if self.encoder_type == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(resnet.children())[:-1])
            in_features = resnet.fc.in_features
            
        elif self.encoder_type == "efficientnet_b0":
            eff = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(eff.children())[:-1])
            in_features = eff.classifier[1].in_features
            
        elif self.encoder_type == "efficientnet_b1":
            eff = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(eff.children())[:-1])
            in_features = eff.classifier[1].in_features
            
        elif self.encoder_type == "vit_b_16":
            vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            in_features = vit.heads.head.in_features
            vit.heads = nn.Identity()
            self.backbone = vit
            
        elif self.encoder_type == "convnext_t":
            cnxt = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(cnxt.children())[:-1])
            in_features = cnxt.classifier[2].in_features
            
        elif self.encoder_type == "swin_t":
            swin = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
            in_features = swin.head.in_features
            swin.head = nn.Identity()
            self.backbone = swin
        
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {encoder_type}")

        self.linear = nn.Linear(in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(images)
        
        if "vit" in self.encoder_type or "swin" in self.encoder_type:
            features = features
        else:
            features = features.view(features.size(0), -1)
            
        features = self.bn(self.linear(features))
        return features

# 2. 맞춤형 Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, decoder_type="lstm"):
        super(DecoderRNN, self).__init__()
        self.decoder_type = decoder_type.lower()
        self.embed = nn.Embedding(vocab_size, embed_size)
        
        # 모델 타입에 따라 다른 RNN 셀 선택
        if self.decoder_type == "lstm":
            self.rnn = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        elif self.decoder_type == "gru":
            self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError(f"지원하지 않는 디코더 타입: {decoder_type}")
            
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.rnn(embeddings)
        outputs = self.linear(hiddens)
        return outputs

# 3. Full Model
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, encoder_type="resnet50", decoder_type="lstm"):
        super(CNNtoRNN, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoderCNN = EncoderCNN(embed_size, encoder_type=encoder_type)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, decoder_type=decoder_type)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs