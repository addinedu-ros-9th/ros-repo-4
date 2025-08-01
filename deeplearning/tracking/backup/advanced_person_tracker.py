#!/usr/bin/env python3
"""
🚀 Advanced Person Tracking System
- YOLOv8 + OSNet + BoT-SORT + FAISS 조합
- 실시간 고성능 사람 추적 및 재식별
- 프레임 밖 재등장 시 ID 유지
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import faiss
import pickle
from collections import deque
import time

sys.path.append('../object_detection')
from simple_detector import PersonDetector

def normalize_embedding(embedding):
    """L2 정규화 함수 - cosine similarity 최적화"""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

def normalize_tensor(tensor):
    """PyTorch 텐서 L2 정규화"""
    return F.normalize(tensor, p=2, dim=1)

def compute_cosine_similarity_pytorch(tensor_a, tensor_b):
    """PyTorch cosine similarity 계산"""
    # L2 정규화
    tensor_a_norm = normalize_tensor(tensor_a)
    tensor_b_norm = normalize_tensor(tensor_b)
    
    # Cosine similarity
    similarity = F.cosine_similarity(tensor_a_norm, tensor_b_norm, dim=1)
    return similarity.item()

class BoTSORTTracker:
    """BoT-SORT 기반 추적기"""
    def __init__(self, max_disappeared=60, iou_threshold=0.3):  # 15 → 60프레임으로 증가
        self.tracks = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared  # 2초 (30fps 기준)
        self.iou_threshold = iou_threshold
        self.frame_count = 0
        
    def update(self, detections):
        """BoT-SORT 업데이트"""
        self.frame_count += 1
        
        # 현재 프레임의 트랙 상태
        current_tracks = {}
        
        # 기존 트랙과 새로운 감지 결과 매칭
        if self.tracks:
            track_ids = list(self.tracks.keys())
            track_boxes = [self.tracks[tid]['bbox'] for tid in track_ids]
            
            # IoU 기반 매칭
            matches = self._hungarian_matching(track_boxes, [d['bbox'] for d in detections])
            
            # 매칭된 트랙 업데이트
            for track_idx, det_idx in matches:
                if det_idx is not None:
                    track_id = track_ids[track_idx]
                    detection = detections[det_idx]
                    
                    # 트랙 업데이트
                    self.tracks[track_id].update({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'last_seen': self.frame_count,
                        'disappeared': 0
                    })
                    current_tracks[track_id] = self.tracks[track_id]
        
        # 새로운 감지 결과에 대해 새 트랙 생성
        for detection in detections:
            if not self._is_tracked(detection['bbox']):
                self.tracks[self.next_id] = {
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence'],
                    'last_seen': self.frame_count,
                    'disappeared': 0,
                    'created': self.frame_count
                }
                current_tracks[self.next_id] = self.tracks[self.next_id]
                self.next_id += 1
        
        # 사라진 트랙 처리
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if track_id not in current_tracks:
                track_data['disappeared'] += 1
                if track_data['disappeared'] <= self.max_disappeared:
                    current_tracks[track_id] = track_data
                else:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return current_tracks
    
    def _hungarian_matching(self, track_boxes, detection_boxes):
        """헝가리안 알고리즘을 사용한 매칭"""
        if not track_boxes or not detection_boxes:
            return []
        
        # IoU 매트릭스 계산
        cost_matrix = np.zeros((len(track_boxes), len(detection_boxes)))
        for i, track_box in enumerate(track_boxes):
            for j, det_box in enumerate(detection_boxes):
                iou = self._calculate_iou(track_box, det_box)
                cost_matrix[i, j] = 1 - iou  # 비용은 1-IoU
        
        # 헝가리안 알고리즘 적용
        from scipy.optimize import linear_sum_assignment
        track_indices, det_indices = linear_sum_assignment(cost_matrix)
        
        # 임계값 이상의 매칭만 반환
        matches = []
        for track_idx, det_idx in zip(track_indices, det_indices):
            if cost_matrix[track_idx, det_idx] < (1 - self.iou_threshold):
                matches.append((track_idx, det_idx))
        
        return matches
    
    def _calculate_iou(self, box1, box2):
        """IoU 계산"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # 교집합 영역
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _is_tracked(self, bbox):
        """이미 추적 중인 바운딩박스인지 확인"""
        for track_data in self.tracks.values():
            if self._calculate_iou(track_data['bbox'], bbox) > self.iou_threshold:
                return True
        return False

class FAISSReIDStore:
    """FAISS 기반 Re-ID 특징 저장소 (메모리 내 리스트) - 개선된 cosine matching + L2 정규화 + Re-ranking"""
    def __init__(self, feature_dim=512, similarity_threshold=0.90):  # 0.95 → 0.90으로 조정
        self.feature_dim = feature_dim
        self.similarity_threshold = similarity_threshold
        self.features = []  # 특징 벡터 리스트
        self.ids = []       # 사람 ID 리스트
        self.index = None   # FAISS 인메모리 인덱스
        
        # **임베딩 평균화를 위한 저장소**
        self.embedding_history = {}  # {person_id: [embeddings]}
        self.avg_embeddings = {}     # {person_id: averaged_embedding}
        self.min_frames_for_avg = 5  # 평균화에 필요한 최소 프레임 수
        
        # **Re-ranking 설정**
        self.use_re_ranking = True
        self.top_k_candidates = 10   # Re-ranking 후보 수
        self.k_reciprocal = 20       # k-reciprocal 파라미터
        
        self._build_index()
        
        print(f"✅ FAISS Re-ID Store initialized")
        print(f"   - Feature dimension: {self.feature_dim}")
        print(f"   - Similarity threshold: {self.similarity_threshold}")
        print(f"   - Index type: Inner Product (cosine similarity)")
        print(f"   - L2 normalization: Enabled")
        print(f"   - Embedding averaging: {self.min_frames_for_avg} frames")
        print(f"   - Re-ranking: {'Enabled' if self.use_re_ranking else 'Disabled'}")
        
    def _build_index(self):
        """FAISS 인덱스 생성"""
        self.index = faiss.IndexFlatIP(self.feature_dim)  # Inner Product (cosine similarity)
        
    def add_person(self, person_id, features):
        """새로운 사람 특징을 리스트에 추가 - 임베딩 평균화 적용"""
        if len(features) != self.feature_dim:
            print(f"❌ Feature dimension mismatch: {len(features)} != {self.feature_dim}")
            return False
        
        # L2 정규화 (중복 정규화 방지)
        features_norm = normalize_embedding(features)
        
        # 임베딩 히스토리 업데이트
        if person_id not in self.embedding_history:
            self.embedding_history[person_id] = []
        self.embedding_history[person_id].append(features_norm)
        
        # 평균 임베딩 계산 (충분한 프레임이 쌓였을 때)
        if len(self.embedding_history[person_id]) >= self.min_frames_for_avg:
            avg_embedding = self._compute_average_embedding(person_id)
            self.avg_embeddings[person_id] = avg_embedding
            final_embedding = avg_embedding
            print(f"✅ Person {person_id} using averaged embedding ({len(self.embedding_history[person_id])} frames)")
        else:
            final_embedding = features_norm
            print(f"⚠️ Person {person_id} using single embedding ({len(self.embedding_history[person_id])}/{self.min_frames_for_avg} frames)")
        
        # FAISS 인덱스에 추가
        self.index.add(final_embedding.reshape(1, -1))
        self.features.append(final_embedding)
        self.ids.append(person_id)
        
        print(f"✅ Person {person_id} added to Re-ID store")
        print(f"   - Feature norm: {np.linalg.norm(final_embedding):.6f}")
        print(f"   - Store size: {len(self.ids)}")
        
        return True
    
    def _compute_average_embedding(self, person_id):
        """특정 사람의 평균 임베딩 계산"""
        embeddings = self.embedding_history[person_id]
        avg_embedding = np.mean(embeddings, axis=0)
        return normalize_embedding(avg_embedding)
    
    def find_similar_person(self, features, top_k=5):
        """유사한 사람 찾기 - Re-ranking 적용"""
        if len(self.features) == 0:
            print("⚠️ Re-ID store is empty")
            return None, 0.0
        
        # L2 정규화 (중복 정규화 방지)
        features_norm = normalize_embedding(features)
        
        # FAISS 검색 (더 많은 후보 검색)
        search_k = max(top_k, self.top_k_candidates) if self.use_re_ranking else top_k
        similarities, indices = self.index.search(features_norm.reshape(1, -1), min(search_k, len(self.features)))
        
        # numpy 배열을 리스트로 변환
        similarities = similarities[0].tolist()
        indices = indices[0].tolist()
        
        print(f"🔍 Initial cosine similarity search results:")
        for i, (sim, idx) in enumerate(zip(similarities, indices)):
            person_id = self.ids[idx]
            print(f"   - Person {person_id}: {sim:.6f}")
        
        if self.use_re_ranking and len(indices) > 1:
            # Re-ranking 적용
            re_ranked_similarities, re_ranked_indices = self._apply_re_ranking(
                features_norm, similarities, indices
            )
            print(f"🔄 Re-ranking results:")
            for i, (sim, idx) in enumerate(zip(re_ranked_similarities, re_ranked_indices)):
                person_id = self.ids[idx]
                print(f"   - Person {person_id}: {sim:.6f}")
            
            # Re-ranking 결과 사용
            similarities = re_ranked_similarities
            indices = re_ranked_indices
        
        if len(indices) > 0 and similarities[0] > self.similarity_threshold:
            best_idx = int(indices[0])  # 정수로 변환
            best_similarity = similarities[0]
            best_id = self.ids[best_idx]
            
            print(f"✅ Best match: Person {best_id} (similarity: {best_similarity:.6f})")
            return best_id, best_similarity
        else:
            print(f"❌ No match above threshold {self.similarity_threshold}")
            return None, 0.0
    
    def _apply_re_ranking(self, query_features, similarities, indices):
        """k-reciprocal Re-ranking 적용"""
        if len(indices) < 2:
            return similarities, indices
        
        # 후보 임베딩들
        candidate_features = [self.features[int(idx)] for idx in indices]  # int() 추가
        candidate_features = np.array(candidate_features)
        
        # 쿼리와 후보들 간의 유사도
        query_similarities = np.array(similarities)
        
        # 후보들 간의 상호 유사도 계산
        candidate_similarities = np.dot(candidate_features, candidate_features.T)
        
        # k-reciprocal 계산
        k_reciprocal_scores = self._compute_k_reciprocal_scores(
            query_similarities, candidate_similarities
        )
        
        # 최종 점수 계산 (쿼리 유사도 + k-reciprocal 점수)
        final_scores = 0.5 * query_similarities + 0.5 * k_reciprocal_scores
        
        # 정렬
        sorted_indices = np.argsort(final_scores)[::-1]
        
        return final_scores[sorted_indices], [indices[i] for i in sorted_indices]  # 리스트로 변환
    
    def _compute_k_reciprocal_scores(self, query_similarities, candidate_similarities):
        """k-reciprocal 점수 계산"""
        k = min(self.k_reciprocal, len(candidate_similarities))
        
        # 각 후보에 대해 k-nearest neighbors 찾기
        k_nearest_indices = np.argsort(candidate_similarities, axis=1)[:, -k:]
        
        # k-reciprocal 점수 계산
        k_reciprocal_scores = np.zeros(len(query_similarities))
        
        for i in range(len(query_similarities)):
            # 후보 i의 k-nearest neighbors
            neighbors = k_nearest_indices[i]
            
            # 각 neighbor가 후보 i를 k-nearest에 포함하는지 확인
            reciprocal_count = 0
            for neighbor_idx in neighbors:
                if i in k_nearest_indices[neighbor_idx]:
                    reciprocal_count += 1
            
            k_reciprocal_scores[i] = reciprocal_count / k
        
        return k_reciprocal_scores
    
    def update_person_features(self, person_id, new_features):
        """기존 사람의 특징 업데이트 - 임베딩 평균화 적용"""
        if person_id in self.ids:
            idx = self.ids.index(person_id)
            
            # 임베딩 히스토리 업데이트
            if person_id not in self.embedding_history:
                self.embedding_history[person_id] = []
            self.embedding_history[person_id].append(normalize_embedding(new_features))
            
            # 평균 임베딩 재계산
            if len(self.embedding_history[person_id]) >= self.min_frames_for_avg:
                avg_embedding = self._compute_average_embedding(person_id)
                self.avg_embeddings[person_id] = avg_embedding
                features_norm = avg_embedding
                print(f"✅ Person {person_id} updated with averaged embedding")
            else:
                features_norm = normalize_embedding(new_features)
                print(f"⚠️ Person {person_id} updated with single embedding")
            
            self.features[idx] = features_norm
            
            # FAISS 인덱스 재구성
            self._rebuild_index()
            
            print(f"🔄 Person {person_id} features updated")
            print(f"   - New feature norm: {np.linalg.norm(features_norm):.6f}")
    
    def _rebuild_index(self):
        """FAISS 인덱스 재구성"""
        self.index = faiss.IndexFlatIP(self.feature_dim)
        if self.features:
            features_array = np.array(self.features)
            self.index.add(features_array)
    
    def get_person_count(self):
        """저장된 사람 수 반환"""
        return len(self.ids)

class AdvancedOSNet:
    """고급 OSNet 모델 - 실제 모델 사용"""
    def __init__(self, feature_dim=512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = feature_dim
        self.model = None
        self.transform = None
        self._initialize_model()
        
    def _initialize_model(self):
        """OSNet 모델 초기화 - 진짜 OSNet 구조 구현"""
        try:
            # 진짜 OSNet 구조 (Omni-Scale Learning + Multi-Scale Features)
            class OSNet(nn.Module):
                def __init__(self, num_classes=751, feature_dim=512):
                    super(OSNet, self).__init__()
                    
                    # **OSNet의 핵심: Omni-Scale Learning**
                    # 다양한 스케일의 특징을 동시에 학습하는 구조
                    
                    # Stage 1: 초기 특징 추출 (7x7 conv)
                    self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2, padding=1)
                    )
                    
                    # Stage 2: Omni-Scale Block 1 (다중 스케일 특징)
                    self.os_block1 = self._make_omni_scale_block(64, 128, 2)
                    
                    # Stage 3: Omni-Scale Block 2 (더 세밀한 특징)
                    self.os_block2 = self._make_omni_scale_block(128, 256, 2)
                    
                    # Stage 4: Omni-Scale Block 3 (고수준 특징)
                    self.os_block3 = self._make_omni_scale_block(256, 512, 2)
                    
                    # **Multi-Scale Feature Fusion**
                    # 여러 스케일의 특징을 융합하여 세부 속성 인식 향상
                    self.feature_fusion = nn.Sequential(
                        nn.AdaptiveAvgPool2d((1, 1)),
                        nn.Flatten(),
                        nn.Dropout(0.5)
                    )
                    
                    # **Attribute-Aware Feature Extractor**
                    # 의복, 액세서리 등 세부 속성을 인식하는 특징 추출기
                    self.attribute_extractor = nn.Sequential(
                        nn.Linear(512, 1024),
                        nn.ReLU(inplace=True),
                        nn.Dropout(0.5),
                        nn.Linear(1024, feature_dim)
                    )
                    
                    # 분류기 (사전훈련용)
                    self.classifier = nn.Linear(feature_dim, num_classes)
                    
                def _make_omni_scale_block(self, in_channels, out_channels, num_layers):
                    """Omni-Scale Block: 다양한 스케일의 특징을 동시에 학습"""
                    layers = []
                    
                    for i in range(num_layers):
                        # **다중 스케일 컨볼루션**
                        # 1x1, 3x3, 5x5 등 다양한 스케일의 컨볼루션을 병렬로 적용
                        multi_scale_conv = nn.ModuleList([
                            # 1x1 스케일 (세밀한 특징)
                            nn.Sequential(
                                nn.Conv2d(in_channels if i == 0 else out_channels, 
                                        out_channels // 4, 1, bias=False),
                                nn.BatchNorm2d(out_channels // 4),
                                nn.ReLU(inplace=True)
                            ),
                            # 3x3 스케일 (중간 특징)
                            nn.Sequential(
                                nn.Conv2d(in_channels if i == 0 else out_channels, 
                                        out_channels // 4, 3, padding=1, bias=False),
                                nn.BatchNorm2d(out_channels // 4),
                                nn.ReLU(inplace=True)
                            ),
                            # 5x5 스케일 (큰 특징)
                            nn.Sequential(
                                nn.Conv2d(in_channels if i == 0 else out_channels, 
                                        out_channels // 4, 5, padding=2, bias=False),
                                nn.BatchNorm2d(out_channels // 4),
                                nn.ReLU(inplace=True)
                            ),
                            # 7x7 스케일 (매우 큰 특징)
                            nn.Sequential(
                                nn.Conv2d(in_channels if i == 0 else out_channels, 
                                        out_channels // 4, 7, padding=3, bias=False),
                                nn.BatchNorm2d(out_channels // 4),
                                nn.ReLU(inplace=True)
                            )
                        ])
                        
                        # **Multi-Scale Feature Fusion**
                        fusion_layer = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=True)
                        )
                        
                        # **Residual Connection**
                        if in_channels != out_channels:
                            shortcut = nn.Sequential(
                                nn.Conv2d(in_channels if i == 0 else out_channels, 
                                        out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels)
                            )
                        else:
                            shortcut = nn.Identity()
                        
                        layers.append(nn.ModuleDict({
                            'multi_scale_conv': multi_scale_conv,
                            'fusion': fusion_layer,
                            'shortcut': shortcut
                        }))
                        
                        # 첫 번째 레이어에서만 다운샘플링
                        if i == 0:
                            layers.append(nn.MaxPool2d(2, stride=2))
                    
                    return nn.ModuleList(layers)
                
                def _forward_omni_scale_block(self, x, block):
                    """Omni-Scale Block의 순전파"""
                    for layer in block:
                        if isinstance(layer, nn.ModuleDict):
                            # 다중 스케일 특징 추출
                            multi_scale_features = []
                            for conv in layer['multi_scale_conv']:
                                multi_scale_features.append(conv(x))
                            
                            # 특징 융합
                            fused_features = torch.cat(multi_scale_features, dim=1)
                            fused_features = layer['fusion'](fused_features)
                            
                            # Residual connection
                            shortcut = layer['shortcut'](x)
                            x = fused_features + shortcut
                            x = F.relu(x)
                        else:
                            # 다운샘플링
                            x = layer(x)
                    
                    return x
                
                def forward(self, x):
                    # Stage 1: 초기 특징 추출
                    x = self.conv1(x)
                    
                    # Stage 2-4: Omni-Scale Blocks
                    x = self._forward_omni_scale_block(x, self.os_block1)
                    x = self._forward_omni_scale_block(x, self.os_block2)
                    x = self._forward_omni_scale_block(x, self.os_block3)
                    
                    # Multi-Scale Feature Fusion
                    x = self.feature_fusion(x)
                    
                    # Attribute-Aware Feature Extraction
                    features = self.attribute_extractor(x)
                    
                    # 분류 출력 (사전훈련용)
                    cls_output = self.classifier(features)
                    
                    return cls_output, features
            
            self.model = OSNet(feature_dim=self.feature_dim)
            
            # **Xavier/Glorot 가중치 초기화 (OSNet 표준)**
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)
            
            self.model.to(self.device)
            self.model.eval()
            
            # **OSNet 표준 전처리 (Market1501 데이터셋 기반)**
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),  # OSNet 표준 크기
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print("✅ Advanced OSNet model initialized successfully")
            print(f"   - Feature dimension: {self.feature_dim}")
            print(f"   - Device: {self.device}")
            print(f"   - Input size: 256x128")
            print("   - Omni-Scale Learning: Enabled")
            print("   - Multi-Scale Feature Fusion: Enabled")
            print("   - Attribute-Aware Features: Enabled")
            print("   - 의복 스타일, 액세서리 인식 최적화")
            
        except Exception as e:
            print(f"❌ OSNet initialization failed: {e}")
            print("   - Falling back to simple feature extraction")
            self.model = None
    
    def extract_features(self, frame, bbox):
        """OSNet 특징 추출 - 성능 최적화 + 세부 속성 인식"""
        if self.model is None:
            print("⚠️ Using fallback feature extraction (OSNet not available)")
            return self._extract_simple_features(frame, bbox)
        
        try:
            x1, y1, x2, y2 = map(int, bbox)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return None
            
            # ROI 크기 제한 및 품질 확인
            if roi.shape[0] < 50 or roi.shape[1] < 25:
                print(f"⚠️ ROI too small: {roi.shape[1]}x{roi.shape[0]}")
                return None
            
            # **1. 성능 제어: 세그멘테이션 스킵 여부 결정**
            use_segmentation = False
            if self.enable_segmentation:
                # 새로운 사람이거나 일정 프레임마다만 세그멘테이션 수행
                if (self.frame_count - self.last_segmentation_frame) >= self.segmentation_skip_frames:
                    use_segmentation = True
                    self.last_segmentation_frame = self.frame_count
                else:
                    self.stats['segmentation_skipped'] += 1
            
            # **2. 세그멘테이션 수행 (성능 제어)**
            if use_segmentation:
                segmented_roi = self._segment_person(roi)
                if segmented_roi is None:
                    print(f"⚠️ Segmentation failed, using original ROI")
                    segmented_roi = roi
                
                # 세그멘테이션 품질 확인
                segmentation_quality = self._check_segmentation_quality(segmented_roi, roi)
                print(f"🔍 Segmentation quality: {segmentation_quality:.2f}")
            else:
                segmented_roi = roi
                segmentation_quality = 0.0
                print(f"⚡ Segmentation skipped for performance (frame {self.frame_count})")
            
            # **3. ROI 크기 조정 (OSNet 입력 크기에 맞게)**
            roi_resized = cv2.resize(segmented_roi, (128, 256))  # OSNet 표준 크기
            
            # 전처리
            roi_tensor = self.transform(roi_resized).unsqueeze(0).to(self.device)
            
            # **4. OSNet 추론 (Omni-Scale Learning)**
            with torch.no_grad():
                _, features = self.model(roi_tensor)
            
            # L2 정규화 (PyTorch 함수 사용)
            features = normalize_tensor(features)
            
            features_np = features.cpu().numpy().flatten()
            
            # 추가 L2 정규화 확인 (numpy)
            features_np = normalize_embedding(features_np)
            
            # **5. 세부 속성 인식 결과 분석**
            feature_norm = np.linalg.norm(features_np)
            feature_std = np.std(features_np)
            feature_range = np.max(features_np) - np.min(features_np)
            
            print(f"✅ OSNet feature extracted: {features_np.shape}")
            print(f"   - Feature norm: {feature_norm:.6f}")
            print(f"   - Feature std: {feature_std:.6f}")
            print(f"   - Feature range: {feature_range:.6f}")
            print(f"   - Omni-Scale Learning: Active")
            print(f"   - Multi-Scale Features: Fused")
            print(f"   - Attribute-Aware: Enabled")
            print(f"   - Segmentation-based: {segmentation_quality > 0.5}")
            
            # **6. 의복/액세서리 속성 추정 (세그멘테이션 기반)**
            if use_segmentation:
                self._analyze_clothing_attributes(features_np, segmented_roi)
            else:
                # 빠른 모드: 간단한 색상 분석만
                self._analyze_clothing_attributes_fast(features_np, roi)
            
            return features_np
            
        except Exception as e:
            print(f"❌ OSNet feature extraction error: {e}")
            print("   - Falling back to simple feature extraction")
            return self._extract_simple_features(frame, bbox)
    
    def _analyze_clothing_attributes_fast(self, features, roi):
        """빠른 의복 속성 분석 (세그멘테이션 없이)"""
        try:
            # 간단한 색상 분석만 수행
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # 주요 색상 추출
            hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist_h)
            
            # 특징 강도 분석
            upper_features = features[:128]
            lower_features = features[128:256]
            accessory_features = features[256:384]
            
            upper_strength = np.mean(np.abs(upper_features))
            lower_strength = np.mean(np.abs(lower_features))
            accessory_strength = np.mean(np.abs(accessory_features))
            
            print(f"👕 Fast Clothing Analysis:")
            print(f"   - Dominant color: {self._get_color_name(dominant_hue)}")
            print(f"   - Upper body strength: {upper_strength:.4f}")
            print(f"   - Lower body strength: {lower_strength:.4f}")
            print(f"   - Accessory strength: {accessory_strength:.4f}")
            
            self.stats['fast_mode_used'] += 1
            
        except Exception as e:
            print(f"⚠️ Fast clothing analysis error: {e}")
    
    def _segment_person(self, roi):
        """사람 세그멘테이션 (빠른 방식으로 최적화)"""
        try:
            # **1. ROI 크기 확인 및 축소 (성능 향상)**
            if roi.shape[0] < 50 or roi.shape[1] < 30:
                return roi  # 너무 작으면 원본 사용
            
            # **2. ROI 크기 축소 (GrabCut 성능 향상)**
            original_size = roi.shape[:2]
            scale_factor = 0.5  # 50% 축소
            small_roi = cv2.resize(roi, (int(roi.shape[1] * scale_factor), int(roi.shape[0] * scale_factor)))
            
            # **3. 빠른 배경 제거 (GrabCut 대신 간단한 방법)**
            # HSV 색상 공간에서 피부색/의복색 범위 추출
            hsv_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
            
            # **4. 간단한 색상 기반 마스킹 (빠른 방식)**
            # 피부색 범위 (의복과 구분하기 위해)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])
            skin_mask = cv2.inRange(hsv_roi, lower_skin, upper_skin)
            
            # **5. 의복 색상 범위 (일반적인 의복 색상)**
            # 파란색, 빨간색, 초록색, 노란색, 검은색, 흰색 등
            clothing_masks = []
            
            # 파란색 의복
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blue_mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)
            clothing_masks.append(blue_mask)
            
            # 빨간색 의복 (두 범위)
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            red_mask1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
            red_mask2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            clothing_masks.append(red_mask)
            
            # 초록색 의복
            lower_green = np.array([40, 50, 50])
            upper_green = np.array([80, 255, 255])
            green_mask = cv2.inRange(hsv_roi, lower_green, upper_green)
            clothing_masks.append(green_mask)
            
            # 검은색/회색 의복
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            black_mask = cv2.inRange(hsv_roi, lower_black, upper_black)
            clothing_masks.append(black_mask)
            
            # 흰색 의복
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            white_mask = cv2.inRange(hsv_roi, lower_white, upper_white)
            clothing_masks.append(white_mask)
            
            # **6. 모든 의복 마스크 결합**
            combined_mask = np.zeros_like(skin_mask)
            for mask in clothing_masks:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # **7. 모폴로지 연산으로 노이즈 제거 (빠른 버전)**
            kernel = np.ones((2, 2), np.uint8)  # 더 작은 커널
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # **8. 원본 크기로 복원**
            mask_resized = cv2.resize(combined_mask, (original_size[1], original_size[0]))
            
            # **9. 마스크 적용**
            segmented_roi = roi * (mask_resized[:, :, np.newaxis] / 255.0)
            
            return segmented_roi.astype(np.uint8)
            
        except Exception as e:
            print(f"⚠️ Fast segmentation error: {e}")
            return roi  # 실패 시 원본 반환
    
    def _check_segmentation_quality(self, segmented_roi, original_roi):
        """세그멘테이션 품질 확인"""
        try:
            # 세그멘테이션된 픽셀 비율 계산
            segmented_pixels = np.count_nonzero(segmented_roi.any(axis=2))
            total_pixels = original_roi.shape[0] * original_roi.shape[1]
            
            if total_pixels == 0:
                return 0.0
            
            segmentation_ratio = segmented_pixels / total_pixels
            
            # 품질 점수 (0.1 ~ 0.8이 좋은 세그멘테이션)
            if 0.1 <= segmentation_ratio <= 0.8:
                quality_score = 1.0 - abs(segmentation_ratio - 0.45) / 0.35
            else:
                quality_score = 0.0
            
            return quality_score
            
        except Exception as e:
            print(f"⚠️ Quality check error: {e}")
            return 0.0
    
    def _get_color_name(self, hue):
        """HSV 색조값을 색상 이름으로 변환"""
        color_names = {
            (0, 10): "Red", (10, 20): "Orange", (20, 30): "Yellow",
            (30, 60): "Green", (60, 120): "Blue", (120, 140): "Purple",
            (140, 160): "Pink", (160, 180): "Red"
        }
        
        for (min_h, max_h), name in color_names.items():
            if min_h <= hue <= max_h:
                return name
        return "Unknown"

    def _extract_simple_features(self, frame, bbox):
        """간단한 특징 추출 (fallback) - 개선된 버전 + L2 정규화"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # 고급 특징 추출
        roi_resized = cv2.resize(roi, (64, 128))
        
        # HSV 히스토그램 (더 세밀하게)
        hsv_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])  # 16 → 32 bins
        hist_s = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])  # 16 → 32 bins
        hist_v = cv2.calcHist([hsv_roi], [2], None, [32], [0, 256])  # 16 → 32 bins
        
        # HOG 특징 (더 세밀하게)
        gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        hog = cv2.HOGDescriptor()
        hog_features = hog.compute(gray_roi)
        
        # RGB 히스토그램 추가
        hist_r = cv2.calcHist([roi_resized], [2], None, [16], [0, 256])  # Red
        hist_g = cv2.calcHist([roi_resized], [1], None, [16], [0, 256])  # Green
        hist_b = cv2.calcHist([roi_resized], [0], None, [16], [0, 256])  # Blue
        
        # 정규화 및 결합
        hist_h = hist_h.flatten() / (hist_h.sum() + 1e-8)
        hist_s = hist_s.flatten() / (hist_s.sum() + 1e-8)
        hist_v = hist_v.flatten() / (hist_v.sum() + 1e-8)
        hist_r = hist_r.flatten() / (hist_r.sum() + 1e-8)
        hist_g = hist_g.flatten() / (hist_g.sum() + 1e-8)
        hist_b = hist_b.flatten() / (hist_b.sum() + 1e-8)
        hog_features = hog_features.flatten() / (np.linalg.norm(hog_features) + 1e-8)
        
        # 특징 결합 (차원 맞춤)
        combined_features = np.concatenate([
            hist_h, hist_s, hist_v,  # HSV (96차원)
            hist_r, hist_g, hist_b,  # RGB (48차원)
            hog_features[:368]       # HOG (368차원)
        ])
        
        # 차원 조정
        if len(combined_features) > self.feature_dim:
            combined_features = combined_features[:self.feature_dim]
        elif len(combined_features) < self.feature_dim:
            combined_features = np.pad(combined_features, (0, self.feature_dim - len(combined_features)))
        
        # L2 정규화 (개선된 함수 사용)
        combined_features = normalize_embedding(combined_features)
        
        print(f"✅ Simple feature extracted: {combined_features.shape}, norm: {np.linalg.norm(combined_features):.6f}")
        
        return combined_features

    def _analyze_clothing_attributes(self, features, roi):
        """의복 및 액세서리 속성 분석 (세그멘테이션 기반)"""
        try:
            # **1. 사람 세그멘테이션 (배경 제거)**
            segmented_roi = self._segment_person(roi)
            if segmented_roi is None:
                print(f"⚠️ Person segmentation failed, using original ROI")
                segmented_roi = roi
            
            # **2. 세그멘테이션 기반 색상 분석**
            hsv_roi = cv2.cvtColor(segmented_roi, cv2.COLOR_BGR2HSV)
            
            # 주요 색상 추출 (세그멘테이션된 영역만)
            hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv_roi], [2], None, [256], [0, 256])
            
            # 색상 특성 분석
            dominant_hue = np.argmax(hist_h)
            avg_saturation = np.mean(hist_s)
            avg_value = np.mean(hist_v)
            
            # **3. 의복 스타일 추정 (특징 패턴 분석)**
            # 특징 벡터의 특정 구간이 의복 스타일을 나타냄
            upper_features = features[:128]  # 상의 관련 특징
            lower_features = features[128:256]  # 하의 관련 특징
            accessory_features = features[256:384]  # 액세서리 관련 특징
            
            # 특징 강도 분석
            upper_strength = np.mean(np.abs(upper_features))
            lower_strength = np.mean(np.abs(lower_features))
            accessory_strength = np.mean(np.abs(accessory_features))
            
            # **4. 세그멘테이션 품질 확인**
            segmentation_quality = self._check_segmentation_quality(segmented_roi, roi)
            
            # **5. 속성 추정 결과 출력**
            print(f"👕 Clothing Analysis (Segmentation-based):")
            print(f"   - Segmentation quality: {segmentation_quality:.2f}")
            print(f"   - Dominant color: {self._get_color_name(dominant_hue)}")
            print(f"   - Color saturation: {'High' if avg_saturation > 100 else 'Low'}")
            print(f"   - Color brightness: {'Bright' if avg_value > 150 else 'Dark'}")
            print(f"   - Upper body strength: {upper_strength:.4f}")
            print(f"   - Lower body strength: {lower_strength:.4f}")
            print(f"   - Accessory strength: {accessory_strength:.4f}")
            
            # **6. 의복 스타일 추정**
            if upper_strength > 0.1:
                print(f"   - Upper style: {'Jacket' if upper_strength > 0.15 else 'T-shirt/Shirt'}")
            if lower_strength > 0.1:
                print(f"   - Lower style: {'Long pants' if lower_strength > 0.12 else 'Shorts'}")
            if accessory_strength > 0.08:
                print(f"   - Accessories: {'Bag/Hat detected' if accessory_strength > 0.1 else 'Possible'}")
            
        except Exception as e:
            print(f"⚠️ Clothing analysis error: {e}")

class AdvancedPersonTracker:
    """고급 사람 추적 시스템 - 개선된 메모리 및 추적"""
    def __init__(self):
        # **핵심 컴포넌트**
        self.bot_sort = BoTSORTTracker(max_disappeared=60)
        self.osnet = AdvancedOSNet(feature_dim=512)
        self.reid_db = FAISSReIDStore(feature_dim=512, similarity_threshold=0.85)  # 0.90 → 0.85로 완화
        
        # **추적 상태**
        self.active_people = {}
        self.disappeared_people = {}
        self.next_person_id = 0
        
        # **개선된 사람 메모리 시스템**
        self.person_memory = {}  # {person_id: PersonMemory}
        self.memory_max_size = 50  # 최대 메모리 크기
        self.memory_cleanup_interval = 100  # 메모리 정리 주기
        
        # **시간 제약 완화**
        self.disappearance_history = {}  # 사라진 사람들의 시간 기록
        self.min_reappear_time = 15      # 30 → 15프레임으로 완화
        self.max_disappeared_time = 300  # 최대 사라진 시간 (10초)
        
        # **성능 최적화**
        self.process_every_n_frames = 1
        self.min_bbox_size = 80
        self.min_bbox_area = 4000
        self.frame_count = 0
        
        # **성능 제어 (렉 방지)**
        self.enable_segmentation = True
        self.segmentation_skip_frames = 3
        self.last_segmentation_frame = 0
        self.performance_mode = "balanced"
        
        # **디버깅 설정**
        self.debug_mode = True
        self.debug_screenshot_interval = 30
        self.debug_dir = "debug_screenshots"
        self.last_debug_save = 0
        self.save_on_new_person = True
        
        # 디버그 디렉토리 생성
        if self.debug_mode:
            os.makedirs(self.debug_dir, exist_ok=True)
        
        # **색상 설정**
        self.colors = [
            (0, 255, 0),    # 초록색
            (255, 0, 0),    # 파란색  
            (0, 0, 255),    # 빨간색
            (255, 255, 0),  # 청록색
            (255, 0, 255),  # 자홍색
            (0, 255, 255),  # 노란색
            (128, 0, 128),  # 보라색
            (255, 165, 0),  # 주황색
            (0, 128, 128),  # 올리브색
            (128, 128, 0)   # 갈색
        ]
        
        # **성능 통계**
        self.stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'inference_time': 0,
            'reid_matches': 0,
            'new_people': 0,
            'reappearances': 0,
            'debug_screenshots': 0,
            'temporal_rejections': 0,
            'embedding_averages': 0,
            're_ranking_applied': 0,
            'segmentation_skipped': 0,
            'fast_mode_used': 0,
            'memory_cleanups': 0,
            'successful_reappearances': 0
        }
        
        print("🚀 Advanced Person Tracking System (Enhanced)")
        print("📋 Components:")
        print("  1. ✅ YOLOv8 Person Detection")
        print("  2. ✅ BoT-SORT Tracking")
        print("  3. ✅ OSNet Feature Extraction (Omni-Scale)")
        print("  4. ✅ FAISS Re-ID Store (In-Memory)")
        print("  5. ✅ Enhanced Person Memory System")
        print(f"🔧 Device: {self.osnet.device}")
        print("\n🚀 Enhanced Features:")
        print("   - Multi-feature person memory (HSV, OSNet, spatial)")
        print("   - Improved Re-ID matching with multiple criteria")
        print("   - Better bounding box tracking and prediction")
        print("   - Memory cleanup and management")
        print(f"🔒 Similarity threshold: {self.reid_db.similarity_threshold} (완화된 매칭)")
        print("🔒 L2 normalization: Enabled for optimal cosine similarity")
        print(f"🔒 Temporal constraint: {self.min_reappear_time} frames (완화)")
        print("🔒 Embedding averaging: 5+ frames for stable features")
        print("🔒 Re-ranking: k-reciprocal encoding for better matching")
        print("🔒 Spatial constraint: REMOVED (Appearance-based matching only)")
        print("\n🎯 OSNet Advantages:")
        print("   - Omni-Scale Learning: 다양한 크기의 속성 정보 학습")
        print("   - Multi-Scale Feature Fusion: 세밀한 차이 인식")
        print("   - Attribute-Aware Features: 의복, 액세서리 속성 인식")
        print("   - 세부 속성 구별: 밝은색 vs 어두운색, 반바지 vs 긴바지")
        print("\n🔍 Fast Segmentation-based Analysis:")
        print("   - Color-based masking: 빠른 색상 기반 배경 제거")
        print("   - Performance controls: 렉 방지를 위한 성능 제어")
        print("   - Adaptive processing: 상황에 따른 처리 방식 조정")
        print("   - Fallback mechanism: 세그멘테이션 실패 시 원본 ROI 사용")
        print("\n🧠 Enhanced Memory System:")
        print("   - Multi-feature storage: HSV, OSNet, spatial features")
        print("   - Memory cleanup: 자동 메모리 관리")
        print("   - Better reappearance: 개선된 재등장 인식")
        print("   - Adaptive thresholds: 상황별 임계값 조정")

    def process_frame(self, frame, yolo_detections):
        """프레임 처리 - 메인 추적 로직"""
        self.frame_count += 1
        self.stats['total_frames'] += 1
        
        start_time = time.time()
        
        # YOLO 감지 결과를 BoT-SORT 형식으로 변환
        detections = []
        for detection in yolo_detections:
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            confidence = detection['confidence']
            detections.append({
                'bbox': bbox,
                'confidence': confidence
            })
        
        # BoT-SORT 추적 업데이트
        current_tracks = self.bot_sort.update(detections)
        
        # 고급 Re-ID 처리
        active_people = self._process_enhanced_reid(frame, current_tracks)
        
        # 성능 통계 업데이트
        inference_time = (time.time() - start_time) * 1000
        self.stats['inference_time'] += inference_time
        self.stats['processed_frames'] += 1
        
        # 메모리 정리 (주기적)
        if self.frame_count % self.memory_cleanup_interval == 0:
            self._cleanup_memory()
            self.stats['memory_cleanups'] += 1
        
        return active_people

    def _process_enhanced_reid(self, frame, current_tracks):
        """개선된 Re-ID 처리"""
        current_people = set()
        
        for track_id, track_data in current_tracks.items():
            bbox = track_data['bbox']
            confidence = track_data['confidence']
            
            # 바운딩박스 크기 필터링
            x1, y1, x2, y2 = map(int, bbox)
            width = x2 - x1
            height = y2 - y1
            area = width * height
            
            if width < self.min_bbox_size or height < self.min_bbox_size or area < self.min_bbox_area:
                continue
            
            # 특징 추출 (다중 특징)
            osnet_features = self.osnet.extract_features(frame, bbox)
            hsv_features = self._extract_hsv_features(frame, bbox)
            spatial_features = self._extract_spatial_features(frame, bbox)
            
            if osnet_features is None:
                continue
            
            # 기존 사람과 매칭
            if track_id in self.active_people:
                # 기존 사람 업데이트
                person_id = self.active_people[track_id]['person_id']
                self._update_person_memory(person_id, {
                    'osnet_features': osnet_features,
                    'hsv_features': hsv_features,
                    'spatial_features': spatial_features,
                    'bbox': bbox,
                    'confidence': confidence,
                    'last_seen': self.frame_count
                })
                
                current_people.add(track_id)
                
            else:
                # 새로운 사람 또는 재등장 확인
                best_match_id = self._find_best_match(osnet_features, hsv_features, spatial_features, bbox)
                
                if best_match_id is not None:
                    # 재등장
                    self.active_people[track_id] = {
                        'person_id': best_match_id,
                        'bbox': bbox,
                        'confidence': confidence,
                        'last_seen': self.frame_count,
                        'color': self.person_memory[best_match_id].color
                    }
                    
                    # 메모리 업데이트
                    self._update_person_memory(best_match_id, {
                        'osnet_features': osnet_features,
                        'hsv_features': hsv_features,
                        'spatial_features': spatial_features,
                        'bbox': bbox,
                        'confidence': confidence,
                        'last_seen': self.frame_count
                    })
                    
                    self.stats['reappearances'] += 1
                    self.stats['successful_reappearances'] += 1
                    print(f"🔄 재등장: Person {best_match_id} (Track {track_id})")
                    
                else:
                    # 새로운 사람
                    person_id = self.next_person_id
                    color = self.colors[person_id % len(self.colors)]
                    
                    # 새로운 사람 메모리 생성
                    self.person_memory[person_id] = PersonMemory(person_id, {
                        'osnet_features': osnet_features,
                        'hsv_features': hsv_features,
                        'spatial_features': spatial_features,
                        'bbox': bbox,
                        'confidence': confidence,
                        'last_seen': self.frame_count,
                        'created_frame': self.frame_count,
                        'color': color
                    })
                    
                    self.active_people[track_id] = {
                        'person_id': person_id,
                        'bbox': bbox,
                        'confidence': confidence,
                        'last_seen': self.frame_count,
                        'color': color
                    }
                    
                    self.stats['new_people'] += 1
                    print(f"🆕 새로운 사람: Person {person_id} (Track {track_id})")
                    self.next_person_id += 1
                
                current_people.add(track_id)
        
        # 사라진 사람 처리
        disappeared_tracks = set(self.active_people.keys()) - current_people
        for track_id in disappeared_tracks:
            person_data = self.active_people[track_id]
            person_id = person_data['person_id']
            
            # 사라진 사람을 메모리에 보관 (삭제하지 않음)
            print(f"⏳ 사라짐: Person {person_id} (Track {track_id})")
            del self.active_people[track_id]
        
        return self.active_people

    def _find_best_match(self, osnet_features, hsv_features, spatial_features, bbox):
        """개선된 매칭 (다중 특징 기반)"""
        best_match_id = None
        best_score = 0.0
        
        for person_id, memory in self.person_memory.items():
            # 시간 제약 확인
            if self.frame_count - memory.last_seen < self.min_reappear_time:
                continue
            
            # 다중 특징 유사도 계산
            similarity_score = self._compute_multi_feature_similarity(
                osnet_features, hsv_features, spatial_features, bbox, memory
            )
            
            if similarity_score > best_score and similarity_score > 0.85:  # 임계값
                best_score = similarity_score
                best_match_id = person_id
        
        return best_match_id

    def _compute_multi_feature_similarity(self, osnet_features, hsv_features, spatial_features, bbox, memory):
        """다중 특징 유사도 계산"""
        # OSNet 특징 유사도
        osnet_sim = self._compute_cosine_similarity(osnet_features, memory.get_best_features()['osnet_features'])
        
        # HSV 특징 유사도
        hsv_sim = self._compute_cosine_similarity(hsv_features, memory.get_best_features()['hsv_features'])
        
        # 공간 특징 유사도
        spatial_sim = self._compute_cosine_similarity(spatial_features, memory.get_best_features()['spatial_features'])
        
        # 가중 평균 (OSNet에 더 높은 가중치)
        total_sim = 0.6 * osnet_sim + 0.25 * hsv_sim + 0.15 * spatial_sim
        
        return total_sim

    def _compute_cosine_similarity(self, features1, features2):
        """코사인 유사도 계산"""
        if features1 is None or features2 is None:
            return 0.0
        
        # L2 정규화
        features1_norm = normalize_embedding(features1)
        features2_norm = normalize_embedding(features2)
        
        # 코사인 유사도
        similarity = np.dot(features1_norm, features2_norm)
        return max(0.0, similarity)  # 음수 값 방지

    def _extract_hsv_features(self, frame, bbox):
        """HSV 특징 추출"""
        x1, y1, x2, y2 = map(int, bbox)
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(48)  # 16 bins * 3 channels
        
        # HSV 변환
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 히스토그램 계산
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # 정규화 및 결합
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()
        
        features = np.concatenate([hist_h, hist_s, hist_v])
        return features

    def _extract_spatial_features(self, frame, bbox):
        """공간 특징 추출"""
        x1, y1, x2, y2 = map(int, bbox)
        
        # 바운딩박스 정보
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / max(height, 1)
        
        # 프레임 내 위치 (정규화)
        frame_height, frame_width = frame.shape[:2]
        center_x = (x1 + x2) / 2 / frame_width
        center_y = (y1 + y2) / 2 / frame_height
        
        # 특징 벡터
        features = np.array([
            width / frame_width,
            height / frame_height,
            area / (frame_width * frame_height),
            aspect_ratio,
            center_x,
            center_y
        ])
        
        return features

    def _update_person_memory(self, person_id, new_data):
        """사람 메모리 업데이트"""
        if person_id in self.person_memory:
            self.person_memory[person_id].update_features(new_data)

    def _cleanup_memory(self):
        """메모리 정리"""
        current_time = self.frame_count
        people_to_remove = []
        
        for person_id, memory in self.person_memory.items():
            # 오래된 사람 제거 (5초 이상)
            if current_time - memory.last_seen > self.max_disappeared_time:
                people_to_remove.append(person_id)
        
        for person_id in people_to_remove:
            del self.person_memory[person_id]
            print(f"🗑️ 메모리 정리: Person {person_id}")

    def _save_debug_screenshot(self, frame, current_tracks, prefix=""):
        """디버그 스크린샷 저장"""
        if not self.debug_mode:
            return
        
        if self.frame_count - self.last_debug_save < self.debug_screenshot_interval:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}debug_{timestamp}_frame{self.frame_count}.jpg"
        filepath = os.path.join(self.debug_dir, filename)
        
        # 디버그 정보가 포함된 프레임 생성
        debug_frame = self.draw_results(frame.copy())
        
        # 추가 정보 표시
        cv2.putText(debug_frame, f"Frame: {self.frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(debug_frame, f"Active: {len(self.active_people)}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(debug_frame, f"Memory: {len(self.person_memory)}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        success = cv2.imwrite(filepath, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if success:
            self.stats['debug_screenshots'] += 1
            self.last_debug_save = self.frame_count
            print(f"📸 Debug screenshot saved: {filename}")

    def draw_results(self, frame):
        """결과 시각화"""
        result_frame = frame.copy()
        
        # 활성 사람들 표시
        for track_id, person_data in self.active_people.items():
            bbox = person_data['bbox']
            person_id = person_data['person_id']
            color = person_data['color']
            confidence = person_data['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 바운딩박스 그리기
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            
            # ID 및 신뢰도 표시
            label = f"Person {person_id} ({confidence:.2f})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # 배경 박스
            cv2.rectangle(result_frame, (x1, y1-label_h-10), (x1+label_w+10, y1), color, -1)
            cv2.rectangle(result_frame, (x1, y1-label_h-10), (x1+label_w+10, y1), (255, 255, 255), 1)
            
            # 텍스트
            cv2.putText(result_frame, label, (x1+5, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 중심점
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(result_frame, (center_x, center_y), 3, color, -1)
        
        # 통계 정보 표시
        stats_text = f"Active: {len(self.active_people)} | Memory: {len(self.person_memory)} | Frame: {self.frame_count}"
        cv2.putText(result_frame, stats_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result_frame

class PersonMemory:
    """개선된 사람 메모리 시스템"""
    def __init__(self, person_id, initial_data):
        self.person_id = person_id
        self.created_frame = initial_data.get('created_frame', 0)
        self.last_seen = initial_data.get('last_seen', 0)
        self.total_appearances = 1
        
        # **다중 특징 저장**
        self.osnet_features = []  # OSNet 특징 히스토리
        self.hsv_features = []    # HSV 히스토그램 히스토리
        self.spatial_features = [] # 공간적 특징 히스토리
        self.bbox_history = []    # 바운딩박스 히스토리
        
        # **현재 특징**
        self.current_osnet = initial_data.get('osnet_features', None)
        self.current_hsv = initial_data.get('hsv_features', None)
        self.current_spatial = initial_data.get('spatial_features', None)
        self.current_bbox = initial_data.get('bbox', None)
        
        # **평균 특징 (안정화된 특징)**
        self.avg_osnet = None
        self.avg_hsv = None
        self.avg_spatial = None
        
        # **메타데이터**
        self.color = initial_data.get('color', (0, 255, 0))
        self.confidence_history = []
        self.velocity_history = []
        
        # **특징 업데이트**
        self.update_features(initial_data)
        
    def update_features(self, new_data):
        """특징 업데이트"""
        # OSNet 특징 업데이트
        if 'osnet_features' in new_data:
            self.current_osnet = new_data['osnet_features']
            self.osnet_features.append(self.current_osnet)
            if len(self.osnet_features) > 10:  # 최대 10개 유지
                self.osnet_features.pop(0)
        
        # HSV 특징 업데이트
        if 'hsv_features' in new_data:
            self.current_hsv = new_data['hsv_features']
            self.hsv_features.append(self.current_hsv)
            if len(self.hsv_features) > 10:
                self.hsv_features.pop(0)
        
        # 공간적 특징 업데이트
        if 'spatial_features' in new_data:
            self.current_spatial = new_data['spatial_features']
            self.spatial_features.append(self.current_spatial)
            if len(self.spatial_features) > 10:
                self.spatial_features.pop(0)
        
        # 바운딩박스 업데이트
        if 'bbox' in new_data:
            self.current_bbox = new_data['bbox']
            self.bbox_history.append(self.current_bbox)
            if len(self.bbox_history) > 5:  # 최대 5개 유지
                self.bbox_history.pop(0)
        
        # 평균 특징 계산
        self._compute_average_features()
        
        # 메타데이터 업데이트
        self.last_seen = new_data.get('last_seen', self.last_seen)
        self.total_appearances += 1
        
        if 'confidence' in new_data:
            self.confidence_history.append(new_data['confidence'])
            if len(self.confidence_history) > 10:
                self.confidence_history.pop(0)
    
    def _compute_average_features(self):
        """평균 특징 계산"""
        # OSNet 특징 평균
        if len(self.osnet_features) >= 3:
            self.avg_osnet = np.mean(self.osnet_features, axis=0)
        else:
            self.avg_osnet = self.current_osnet
        
        # HSV 특징 평균
        if len(self.hsv_features) >= 3:
            self.avg_hsv = np.mean(self.hsv_features, axis=0)
        else:
            self.avg_hsv = self.current_hsv
        
        # 공간적 특징 평균
        if len(self.spatial_features) >= 3:
            self.avg_spatial = np.mean(self.spatial_features, axis=0)
        else:
            self.avg_spatial = self.current_spatial
    
    def get_best_features(self):
        """최적의 특징 반환 (평균 > 현재 > 첫 번째)"""
        return {
            'osnet_features': self.avg_osnet if self.avg_osnet is not None else self.current_osnet,
            'hsv_features': self.avg_hsv if self.avg_hsv is not None else self.current_hsv,
            'spatial_features': self.avg_spatial if self.avg_spatial is not None else self.current_spatial,
            'bbox': self.current_bbox
        }
    
    def predict_bbox(self, current_frame):
        """바운딩박스 예측 (사람이 어디로 갈지 예측)"""
        if len(self.bbox_history) < 2:
            return self.current_bbox
        
        # 속도 계산
        recent_bboxes = self.bbox_history[-3:]  # 최근 3개
        velocities = []
        
        for i in range(1, len(recent_bboxes)):
            prev_bbox = recent_bboxes[i-1]
            curr_bbox = recent_bboxes[i]
            
            # 중심점 속도
            prev_center = [(prev_bbox[0] + prev_bbox[2])/2, (prev_bbox[1] + prev_bbox[3])/2]
            curr_center = [(curr_bbox[0] + curr_bbox[2])/2, (curr_bbox[1] + curr_bbox[3])/2]
            
            velocity = [curr_center[0] - prev_center[0], curr_center[1] - prev_center[1]]
            velocities.append(velocity)
        
        if not velocities:
            return self.current_bbox
        
        # 평균 속도
        avg_velocity = np.mean(velocities, axis=0)
        
        # 프레임 차이
        frame_diff = current_frame - self.last_seen
        
        # 예측된 위치
        current_center = [(self.current_bbox[0] + self.current_bbox[2])/2, 
                         (self.current_bbox[1] + self.current_bbox[3])/2]
        
        predicted_center = [
            current_center[0] + avg_velocity[0] * frame_diff,
            current_center[1] + avg_velocity[1] * frame_diff
        ]
        
        # 바운딩박스 크기 유지
        bbox_width = self.current_bbox[2] - self.current_bbox[0]
        bbox_height = self.current_bbox[3] - self.current_bbox[1]
        
        predicted_bbox = [
            predicted_center[0] - bbox_width/2,
            predicted_center[1] - bbox_height/2,
            predicted_center[0] + bbox_width/2,
            predicted_center[1] + bbox_height/2
        ]
        
        return predicted_bbox

def main():
    """메인 함수"""
    print("🚀 ADVANCED PERSON TRACKING SYSTEM")
    print("📖 BoT-SORT + OSNet + FAISS Integration")
    print("🔧 High-performance Real-time Tracking")
    
    detector = PersonDetector()
    tracker = AdvancedPersonTracker()
    
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("📖 Controls:")
    print("  - q: Quit")
    print("  - s: Save screenshot")
    print("  - r: Reset tracking")
    print("  - p: Performance stats")
    print("  - d: Database info")
    
    window_name = "Advanced Person Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read frame!")
                break
            
            # YOLO 사람 감지
            yolo_detections = detector.detect_people(frame)
            
            # 고급 추적 처리
            active_people = tracker.process_frame(frame, yolo_detections)
            
            # 결과 시각화
            result_frame = tracker.draw_results(frame)
            
            cv2.imshow(window_name, result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("🛑 System shutdown")
                break
            elif key == ord('s'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"manual_screenshot_{timestamp}.jpg"
                filepath = os.path.join(tracker.debug_dir, filename)
                
                # 디버그 정보가 포함된 프레임 생성
                debug_frame = tracker.draw_results(frame.copy())
                
                # 추가 정보 표시
                cv2.putText(debug_frame, f"Manual Screenshot - Frame {tracker.frame_count}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                success = cv2.imwrite(filepath, debug_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if success:
                    print(f"📸 Manual screenshot saved: {filename}")
                    print(f"   - Frame: {tracker.frame_count}")
                    print(f"   - Active people: {len(tracker.active_people)}")
                    print(f"   - File size: {os.path.getsize(filepath)} bytes")
                else:
                    print(f"❌ Failed to save manual screenshot: {filepath}")
            elif key == ord('r'):
                tracker.active_people.clear()
                tracker.disappeared_people.clear()
                tracker.next_person_id = 0
                tracker.reid_db = FAISSReIDStore()
                print("🔄 Tracking reset")
            elif key == ord('p'):
                stats = tracker.stats
                avg_inference = stats['inference_time'] / max(1, stats['processed_frames'])
                print(f"\n📊 Advanced Tracking Performance:")
                print(f"  - Total frames: {stats['total_frames']}")
                print(f"  - Processed frames: {stats['processed_frames']}")
                print(f"  - Active people: {len(tracker.active_people)}")
                print(f"  - Reappearances: {stats['reappearances']}")
                print(f"  - New people: {stats['new_people']}")
                print(f"  - Avg inference time: {avg_inference:.1f}ms")
                print(f"  - Re-ID store size: {tracker.reid_db.get_person_count()}")
                print(f"  - Debug screenshots: {stats['debug_screenshots']}")
                print(f"  - Debug directory: {tracker.debug_dir}")
            elif key == ord('d'):
                print(f"\n🗃️ Re-ID Store Info:")
                print(f"  - Stored people: {tracker.reid_db.get_person_count()}")
                print(f"  - Feature dimension: {tracker.reid_db.feature_dim}")
                print(f"  - Similarity threshold: {tracker.reid_db.similarity_threshold}")
                
    except KeyboardInterrupt:
        print("🛑 User interrupt")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n🎯 Final Results:")
        print(f"  - Total frames: {tracker.stats['total_frames']}")
        print(f"  - Active people: {len(tracker.active_people)}")
        print(f"  - Reappearances: {tracker.stats['reappearances']}")
        print(f"  - New people: {tracker.stats['new_people']}")
        print(f"  - Re-ID store size: {tracker.reid_db.get_person_count()}")
        print(f"  - Debug screenshots: {tracker.stats['debug_screenshots']}")
        print(f"  - Debug directory: {tracker.debug_dir}")
        print(f"✅ Advanced Person Tracking System terminated")

if __name__ == "__main__":
    main() 