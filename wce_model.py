import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torchvision.transforms as transforms
import pdb
import time

import os

    # class MyClass:
    #     def __init__(self):
    #         self.variable = "This is a variable"

    #     def variable(self):
    #         print("This is a method")

    # # Create an instance of MyClass
    # obj = MyClass()

    # # Attempting to call the method directly will result in an error because 'variable' is now a string.
    # method_variable = 'variable'
    # # obj.variable()  # This would raise an error because 'variable' is a string, not a callable.

    # # Use getattr to call the method with the same name as the variable
    # method = getattr(obj, method_variable)  # This will fetch the method, not the variable
    # method()  # Call the method



# PS-DeVCEM Model
class PSDeVCEM(nn.Module): # 1024 , 1
    def __init__(self, lstm_hidden_size=512, num_layers=2, bidirectional=True, num_classes=1):
        super(PSDeVCEM, self).__init__()
        dropout_prob = 0.5
        feature_dim = 2048

        self.dropout = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.batchnorm = nn.BatchNorm1d(feature_dim)
        
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.resnet_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove the last FC layer
        
        # LSTM for temporal feature extraction
        self.lstm = nn.LSTM(input_size=resnet.fc.in_features, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=num_layers, 
                            bidirectional=bidirectional, 
                            batch_first=True)
        
        # Attention layer
        # self.attention = Attention(lstm_hidden_size * 2 if bidirectional else lstm_hidden_size)
        lstm_hidden_size_2 = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        self.attention_seq = nn.Sequential(
            nn.Linear(lstm_hidden_size_2, lstm_hidden_size_2),
            nn.Tanh(),
            nn.Linear(lstm_hidden_size_2, 1)
        )

        self.fc1 = nn.Linear(resnet.fc.in_features, resnet.fc.in_features)
        self.fc2 = nn.Linear(resnet.fc.in_features, resnet.fc.in_features)
        
        # Fully connected layer
        self.fc = nn.Linear(lstm_hidden_size * 2 if bidirectional else lstm_hidden_size, num_classes)
        
        
    def forward(self, x): # b, seq, c, h, w
        batch_size, seq_length, c, h, w = x.size()
        
        # Initialize hidden state for LSTM
        h0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers * (2 if self.lstm.bidirectional else 1), batch_size, self.lstm.hidden_size).to(x.device)
        
        # Extract features for each frame using ResNet50
        features = []
        for i in range(seq_length):
            with torch.no_grad():
                frame_features = self.resnet_encoder(x[:, i, :, :, :]).view(batch_size, -1)
            features.append(frame_features)
        features = torch.stack(features, dim=1)  # Shape: (batch_size, seq_length, feature_dim) feature_dim = 2048

        # print(features.shape)
        # batch norm
        features = features.permute(0, 2, 1)  # Shape: (batch_size, feature_dim, seq_length)
        features = self.batchnorm(features)  # Shape: (batch_size, feature_dim, seq_length)
        features = features.permute(0, 2, 1)  # Shape: back to (batch_size, seq_length, feature_dim)
        features = self.dropout(features)  # Shape: back to (batch_size, seq_length, feature_dim)
        
        # # Apply temporal attention
        # features = features * self.attention(features)[1]
        
        ########## # dropout
        ###### features = self.dropout(features)

        # Pass features through LSTM
        lstm_out, _ = self.lstm(features, (h0, c0))  # Shape: (batch_size, seq_length, hidden_size * num_directions)
        
        # dropout
        lstm_out = self.dropout(lstm_out)
        
        # Apply attention mechanism on LSTM output
        attn_output, attn_weights = self.attention(lstm_out)  # Shape: (batch_size, hidden_size), (batch_size, seq_length, 1)
        
        # dropout
        attn_output = self.dropout(attn_output)

        # Apply self-supervision
        self_supervision_loss = self.SelfSupervision(features[:, :seq_length//2], features[:, seq_length//2:])
        
        # Classification for each frame
        output = self.fc(attn_output)  # Shape: (batch_size, num_classes)
        # output = nn.Sigmoid(output)
        
        return output, attn_weights, self_supervision_loss
    
    def attention(self, lstm_out):
        attn_weights = self.attention_seq(lstm_out)  # Shape: (batch_size, seq_length, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # Shape: (batch_size, seq_length, 1)
        attn_output = torch.sum(lstm_out * attn_weights, dim=1)  # Shape: (batch_size, hidden_size)

        return attn_output, attn_weights
        
    def SelfSupervision(self, x_high, x_low):
        dist = torch.norm(self.fc1(x_high) - self.fc2(x_low), dim=1)
        return dist.mean()



class PSDeVCEM2(nn.Module):
    def __init__(self, num_classes=1, hidden_size=512, attention_size=256):
        super(PSDeVCEM2, self).__init__()
        
        # 1. ResNet50 for feature extraction (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove last layers
        
        # 2. Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 3. Residual Bi-directional LSTM
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=2, 
                            batch_first=True, bidirectional=True)
        
        # 4. Temporal Attention Mechanism
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
        
        # 5. Classification Layer
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()
        
        # 6. Self-Supervision Network (Two-Layered Neural Network)
        self.self_supervised_net = nn.Sequential(
            nn.Linear(2 * hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Output is a single value
        )
    
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.size()
        
        # Initialize list to store frame features
        frame_features = []
        
        # Extract features for each frame
        for t in range(seq_length):
            frame = x[:, t, :, :, :]  # Shape: (batch_size, c, h, w)
            with torch.no_grad():
                conv_features = self.resnet(frame)  # Shape: (batch_size, 2048, H, W)
            pooled_features = self.avgpool(conv_features).view(batch_size, -1)  # Shape: (batch_size, 2048)
            frame_features.append(pooled_features)
        
        # Stack frame features
        features = torch.stack(frame_features, dim=1)  # Shape: (batch_size, seq_length, 2048)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_length, 2 * hidden_size)
        
        # Compute attention weights
        attn_weights = self.attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_length)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Compute context vector (video-level representation)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)  # Shape: (batch_size, 2 * hidden_size)
        
        # Classification output
        logits = self.fc(context_vector)
        outputs = logits
        # outputs = self.sigmoid(logits)
        
        # Self-supervision
        self_supervision_loss = self.self_supervision(lstm_out, attn_weights)
        
        return outputs, attn_weights, self_supervision_loss
    
    def self_supervision(self, lstm_out, attn_weights):
        batch_size, seq_length, hidden_size = lstm_out.size()
        
        # Threshold to divide into positive and negative bags
        threshold = 1.0 / seq_length  # 1/N
        
        # Initialize lists to store embeddings
        positive_embeddings = []
        negative_embeddings = []
        
        for i in range(batch_size):
            # Get attention weights and embeddings for the i-th sample
            attn = attn_weights[i]  # Shape: (seq_length,)
            embeddings = lstm_out[i]  # Shape: (seq_length, hidden_size)
            
            # Positive bag: embeddings with attn > threshold
            positive_mask = attn > threshold
            positive_bag = embeddings[positive_mask]
            
            # Negative bag: embeddings with attn <= threshold
            negative_mask = attn <= threshold
            negative_bag = embeddings[negative_mask]
            
            # Aggregate embeddings by averaging
            if positive_bag.size(0) > 0:
                positive_embedding = positive_bag.mean(dim=0)  # Shape: (hidden_size,)
                positive_embeddings.append(positive_embedding)
            if negative_bag.size(0) > 0:
                negative_embedding = negative_bag.mean(dim=0)  # Shape: (hidden_size,)
                negative_embeddings.append(negative_embedding)
        
        # Prepare training pairs and labels
        embeddings = positive_embeddings + negative_embeddings
        labels = [1] * len(positive_embeddings) + [0] * len(negative_embeddings)
        
        if len(embeddings) < 2:
            # Not enough embeddings to compute self-supervision loss
            return torch.tensor(0.0, requires_grad=True)
        
        # Stack embeddings and labels
        embeddings = torch.stack(embeddings)  # Shape: (num_embeddings, hidden_size)
        labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # Shape: (num_embeddings, 1)
        
        # Pass embeddings through self-supervised network
        outputs = self.self_supervised_net(embeddings)  # Shape: (num_embeddings, 1)
        
        # Compute binary cross-entropy loss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, labels.to(outputs.device))
        
        return loss


# class PSDeVCEM3(nn.Module):
#     def __init__(self, num_classes=1, hidden_size=512, attention_size=256, seq_length=30):
#         super(PSDeVCEM3, self).__init__()

#         # ResNet50 for feature extraction (pretrained on ImageNet)
#         resnet = models.resnet50(pretrained=True)
#         self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Removing the final layers
        
#         # Global Average Pooling
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
#         # Residual Bi-directional LSTM
#         self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=2, 
#                             batch_first=True, bidirectional=True)
        
#         # Attention Mechanism (2-layer attention)
#         self.attention = nn.Sequential(
#             nn.Linear(2 * hidden_size, attention_size),
#             nn.Tanh(),
#             nn.Linear(attention_size, 1)
#         )

#         # Fully connected layer for final classification
#         self.fc = nn.Linear(2 * hidden_size, num_classes)
        
#         # Dropout and Sigmoid for classification output
#         self.dropout = nn.Dropout(p=0.5)
#         self.sigmoid = nn.Sigmoid()

#         # Self-supervised fully connected network (2-layer)
#         self.self_supervised_fc = nn.Sequential(
#             nn.Linear(2 * hidden_size, 256),  # Input size is 2*hidden_size (bidirectional LSTM output)
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )
        
#     def forward(self, x):
#         batch_size, seq_length, c, h, w = x.shape
        
#         # 1. ResNet50 feature extraction for each frame
#         features = []
#         for i in range(seq_length):
#             frame_features = self.resnet(x[:, i, :, :, :])  # Extract features for each frame
#             frame_features = self.avgpool(frame_features).view(batch_size, -1)  # Global average pooling
#             features.append(frame_features)
        
#         features = torch.stack(features, dim=1)  # Shape: (batch_size, seq_length, 2048)
        
#         # 2. Residual Bi-directional LSTM
#         lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_length, 2 * hidden_size)
        
#         # 3. Temporal Attention Mechanism
#         attention_weights = self.attention(lstm_out).squeeze(-1)  # Shape: (batch_size, seq_length)
#         attention_weights = F.softmax(attention_weights, dim=1)  # Softmax over sequence length

#         # 4. Weighted sum of LSTM outputs using attention weights
#         context_vector = torch.sum(attention_weights.unsqueeze(-1) * lstm_out, dim=1)  # Shape: (batch_size, 2 * hidden_size)
        
#         # 5. Classification output
#         output = self.fc(context_vector)  # Shape: (batch_size, num_classes)
#         output = self.sigmoid(output)  # For multi-label classification
        
#         return output, attention_weights, lstm_out  # Return lstm_out for self-supervised learning

#     def self_supervision(self, lstm_out, attention_weights):
#         """
#         Implements the self-supervision using a two-layer fully connected network
#         on positive and negative bags of embeddings based on attention weights.
#         """
#         # Group into positive and negative bags based on attention values
#         positive_bag = lstm_out[attention_weights > 1.0 / lstm_out.size(1)]  # Positive bag: higher attention
#         negative_bag = lstm_out[attention_weights <= 1.0 / lstm_out.size(1)]  # Negative bag: lower attention

#         # Compute mean embeddings for positive and negative bags
#         positive_embedding = torch.mean(positive_bag, dim=0)
#         negative_embedding = torch.mean(negative_bag, dim=0)

#         # Pass through self-supervised fully connected network
#         positive_output = self.self_supervised_fc(positive_embedding)  # Output for positive bag
#         negative_output = self.self_supervised_fc(negative_embedding)  # Output for negative bag

#         # Ground truth: "1" for positive, "0" for negative
#         positive_label = torch.ones_like(positive_output)
#         negative_label = torch.zeros_like(negative_output)

#         # Compute self-supervision loss (Binary Cross-Entropy Loss)
#         self_supervision_loss = F.binary_cross_entropy_with_logits(positive_output, positive_label) + \
#                                 F.binary_cross_entropy_with_logits(negative_output, negative_label)

#         return self_supervision_loss


#     # # Example usage:
#     # model = PSDeVCEM(num_classes=14)
#     # input_data = torch.randn(8, 30, 3, 224, 224)  # Example input: batch of 8 videos, each with 30 frames of 224x224 RGB images
#     # output, attention_weights, lstm_out = model(input_data)

#     # # Self-supervision step
#     # self_supervision_loss = model.self_supervision(lstm_out, attention_weights)
#     # print("Self-supervision loss:", self_supervision_loss.item())








class WTALC(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=1, dropout_prob=0.5, s=8, delta=0.5):
        super(WTALC, self).__init__()
        
        # 1. ResNet50 for feature extraction (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove last layers

        # 2. Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer followed by ReLU and Dropout
        self.fc = nn.Linear(2 * feature_dim, feature_dim)  # for concatenated RGB and Flow features
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Label projection to obtain class-wise activations
        self.label_projection = nn.Linear(feature_dim, num_classes)
        
        # Hyperparameters
        self.num_classes = num_classes
        self.s = s  # Hyperparameter for k-max pooling
        self.delta = delta   # Margin for hinge loss

    def forward(self, x, flow_features):
        
        batch_size, seq_length, c, h, w = x.size()
        
        # Initialize list to store frame features
        frame_features = []
        
        # Extract features for each frame
        for t in range(seq_length):
            frame = x[:, t, :, :, :]  # Shape: (batch_size, c, h, w)
            with torch.no_grad():
                conv_features = self.resnet(frame)  # Shape: (batch_size, 2048, H, W)
            pooled_features = self.avgpool(conv_features).view(batch_size, -1)  # Shape: (batch_size, 2048)
            frame_features.append(pooled_features)
        
        # Stack frame features
        rgb_features = torch.stack(frame_features, dim=1)  # Shape: (batch_size, seq_length, 2048)

        # Concatenate RGB and Optical Flow features
        features = torch.cat((rgb_features, flow_features), dim=-1)  # (batch, length, 2 * feature_dim)
        
        # Fully connected, ReLU, and Dropout
        features = self.fc(features)
        features = self.relu(features)
        features = self.dropout(features)
        
        # Project to label space to obtain class-wise activations
        class_wise_activations = self.label_projection(features)  # (batch_size, seq_len, num_classes)
        
        return class_wise_activations

    def k_max_pooling(self, class_activations):
        """
        Perform k-max pooling along the temporal dimension for each class.
        class_activations has shape (batch_size, seq_len, num_classes).
        """
        batch_size, num_classes, seq_len = class_activations.size()
        k = max(1, seq_len // self.s)
        
        # Perform k-max pooling along the temporal (seq_len) dimension
        topk_vals, _ = torch.topk(class_activations, k=k, dim=1)
        
        # Compute the mean of the top-k values (k-max pooling)
        k_max_pooled = topk_vals.mean(dim=1)  # (batch, num_classes)
        
        return k_max_pooled

    def compute_mill_loss(self, pooled_scores, ground_truth):
        """
        Compute the Multiple Instance Learning Loss (MILL).
        pooled_scores has shape (batch_size, num_classes).
        """
        # Apply softmax to convert scores to probabilities
        probabilities = F.softmax(pooled_scores, dim=1)
        
        # Normalize the ground truth labels
        normalized_gt = ground_truth / ground_truth.sum(dim=1, keepdim=True)
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(probabilities, normalized_gt)
        
        return loss

    def compute_casl_loss(self, high_attention_features, low_attention_features):
        """
        Compute Co-Activity Similarity Loss (CASL) using ranking hinge loss.
        """
        delta = self.delta  # Margin for hinge loss
        
        # Cosine similarity between feature vectors
        def cosine_similarity(f1, f2):
            return F.cosine_similarity(f1, f2, dim=-1)
        
        loss = torch.mean(F.relu(
            cosine_similarity(high_attention_features[0], high_attention_features[1]) -
            cosine_similarity(high_attention_features[0], low_attention_features[1]) + delta
        ))
        
        return loss

    def temporal_attention(self, activations): # (batch_size, seq_len, num_classes)
        """
        Compute temporal attention using softmax over the temporal axis.
        """
        attention_weights = F.softmax(activations, dim=1)  # Softmax over time
        return attention_weights


class STPN2(nn.Module):
    def __init__(self, feature_dim=2048, num_classes=1, attention_size=256):
        super(STPN2, self).__init__()
        
        # 1. ResNet50 for feature extraction (pretrained on ImageNet)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # Remove last layers
        
        # 2. Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Attention Module
        self.attention_fc1 = nn.Linear(feature_dim, attention_size)
        self.attention_fc2 = nn.Linear(attention_size, 1) # 1 or attention_size ?
        
        # Final classification layer
        self.classifier = nn.Linear(feature_dim, num_classes)
        

    def attention_module(self, features):
        """
        Computes class-agnostic attention weights for each segment.
        Args:
            features: tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            attention_weights: tensor of shape (batch_size, seq_len)
        """
        attn = F.relu(self.attention_fc1(features))  # Shape: (batch_size, seq_len, attention_size)
        
        # attn = torch.sigmoid(F.relu(self.attention_fc2(attn))).squeeze(-1)  # Shape: (batch_size, seq_len)
        attn = torch.sigmoid(self.attention_fc2(attn)).squeeze(-1)  # Shape: (batch_size, seq_len)
        
        return attn
    
    def forward(self, x):
        """
        Forward pass of STPN.
        Args:
            features: tensor of shape (batch_size, seq_len, feature_dim)
        Returns:
            video_level_preds: tensor of shape (batch_size, num_classes)
            attention_weights: tensor of shape (batch_size, seq_len)
        """

        batch_size, seq_length, c, h, w = x.size()
        
        # Initialize list to store frame features
        frame_features = []
        
        # Extract features for each frame
        for t in range(seq_length):
            frame = x[:, t, :, :, :]  # Shape: (batch_size, c, h, w)
            with torch.no_grad():
                conv_features = self.resnet(frame)  # Shape: (batch_size, 2048, H, W)
            pooled_features = self.avgpool(conv_features).view(batch_size, -1)  # Shape: (batch_size, 2048)
            frame_features.append(pooled_features)
        
        # Stack frame features
        features = torch.stack(frame_features, dim=1)  # Shape: (batch_size, seq_length, 2048)
        
        # 1. Compute attention weights for each segment
        attention_weights = self.attention_module(features)  # Shape: (batch_size, seq_len)
        
        # 2. Apply attention weights to the features
        weighted_features = attention_weights.unsqueeze(-1) * features  # Shape: (batch_size, seq_len, feature_dim)
        
        # 3. Temporal pooling (weighted average pooling)
        pooled_features = torch.sum(weighted_features, dim=1)  # Shape: (batch_size, feature_dim)

        # 4. Classification (video-level)
        class_scores = torch.sigmoid(self.classifier(pooled_features))  # Shape: (batch_size, num_classes)

        # 5. T-CAM (class-specific activations for each segment)
        # Compute class scores for each segment
        segment_scores = torch.sigmoid(self.classifier(features))  # Shape: (batch_size, seq_len, num_classes)
        tcam = segment_scores * attention_weights.unsqueeze(-1)  # Shape: (batch_size, seq_len, num_classes)
        

        return class_scores, attention_weights, tcam, 
    
        # classification_loss = F.binary_cross_entropy(video_level_preds, target)
        # total_loss = classification_loss + beta * sparsity_loss
    
    

    def compute_loss(self, class_scores, tcam, target, attention_weights, beta=0.1):
        """
        Compute the total loss (classification + sparsity + T-CAM influence)
        """
        # 1. Classification loss based on T-CAM (segment-wise)
        tcam_classification_loss = F.binary_cross_entropy(tcam.mean(dim=1), target)

        # 2. Sparsity loss (L1 regularization on attention weights)
        sparsity_loss = torch.mean(torch.abs(attention_weights))
        
        # Total loss: weighted sum of classification loss and sparsity loss
        total_loss = tcam_classification_loss + beta * sparsity_loss

        return total_loss







def main():
    gpu = 1

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'
        
    # 데이터 변환 (비디오의 각 프레임에 적용)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

####################
    # Example usage
    batch_size = 16
    seq_len = 30
    c = 3
    h = 224
    w = 224
    feature_dim = 2048  # Feature dimension for each segment
    num_classes = 1   # Number of action classes

    # Randomly generated features (RGB and flow)
    videos = torch.randn(batch_size, seq_len, c, h, w).cuda() 
    labels = torch.randn(batch_size, num_classes).cuda() # [batch, 1]
    flow_features = torch.randn(batch_size, seq_len, feature_dim).cuda()
    # labels = labels.unsqueeze(1)

##############################
##############################


    start_time = time.time()

    print("load WTALC")
    # Initialize the W-TALC model
    wtalc_model = WTALC(feature_dim=feature_dim, num_classes=num_classes).cuda()

    # Forward pass
    class_wise_activations = wtalc_model(videos, flow_features)  # (batch_size, seq_len, num_classes)

    # Temporal attention
    attention_weights = wtalc_model.temporal_attention(class_wise_activations)

    print(f'attention_weights {attention_weights.shape}')
    print(f'class_wise_activations {class_wise_activations.shape}')

    # Apply attention to the features (weighted by attention weights)
    high_attention_features = torch.einsum('bts, bts->bts', class_wise_activations, attention_weights)

    # Compute the pooled scores using k-max pooling
    pooled_scores = wtalc_model.k_max_pooling(class_wise_activations)

    # Compute Multiple Instance Learning Loss (MILL)
    mill_loss = wtalc_model.compute_mill_loss(pooled_scores, labels)

    # Compute Co-Activity Similarity Loss (CASL) using high/low attention features
    casl_loss = wtalc_model.compute_casl_loss(high_attention_features, high_attention_features)  # As an example

    # Final loss (combining MILL and CASL)
    final_loss = mill_loss + casl_loss
    print("Final Loss:", final_loss.item())
    print(f' -> {time.time()-start_time:.4f}s ')


#######################

    # STPN 모델 정의
    start_time = time.time()

    print("load STPN")
    stpn_model = STPN2().cuda()
    class_scores, attention_weights, tcam, sparsity_loss = stpn_model(videos) # v_outputs=outputs[0].item(), 
    loss = stpn_model.compute_loss(class_scores, tcam, labels, attention_weights)
    
    print(f'class_scores {class_scores}')
    print(f'tcam {tcam[:,:,0]}')
    print(f'tcam_mean {tcam.mean(dim=1)}')

    # Make class prediction (if probability > 0.5, predict 1; else predict 0)
    # pred = (class_scores > 0.5).float()
    
    print(" class_scores output shape:", class_scores.shape)  # (batch_size, num_classes)
    print(" Attention scores shape:", attention_weights.shape)  # (batch_size, num_frames, 1)
    print(" tcam output shape:", tcam.shape)  # (batch_size, num_classes)
    print(" label output shape:", labels.shape)
    print(" tcam_mean shape:", tcam.mean(dim=1).shape)
    print(f' -> {time.time()-start_time:.4f}s ')



#######################

    start_time = time.time()

    print("load PSDeVCEM")
    psdevcem = PSDeVCEM2().cuda()
    output, self_supervision_loss  = psdevcem(videos)
    print(" output shape:", output.shape)  # (batch_size, num_classes)
    # print(" attn_weights shape:", attn_weights.shape)  # (batch_size, num_frames, 1)
    print(f' -> {time.time()-start_time:.4f}s ')

        


# Test the model with random input
if __name__ == "__main__":
    main()




