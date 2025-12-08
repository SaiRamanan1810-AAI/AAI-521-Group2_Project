import os
import json
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.model import load_efficientnet_b0


class InferencePipeline:
    def __init__(self, plant_checkpoint: str, disease_checkpoints: Dict[str, str], device='cpu', threshold=0.5):
        self.device = torch.device(device)
        self.threshold = threshold
        # load plant model
        self.plant_model = None
        self.plant_meta = None
        if plant_checkpoint:
            self.plant_model = load_efficientnet_b0(num_classes=4, pretrained=False)
            ck = torch.load(plant_checkpoint, map_location=self.device)
            self.plant_model.load_state_dict(ck['model_state_dict'])
            self.plant_model.to(self.device).eval()

        # disease models map species->(model, classes)
        self.disease_models = {}
        for species, ckpath in disease_checkpoints.items():
            if not os.path.exists(ckpath):
                continue
            meta_path = ckpath + '.meta.json'
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            else:
                meta = {'classes': []}

            numc = len(meta.get('classes', [])) or 1
            model = load_efficientnet_b0(num_classes=numc, pretrained=False)
            ck = torch.load(ckpath, map_location=self.device)
            model.load_state_dict(ck['model_state_dict'])
            model.to(self.device).eval()
            self.disease_models[species] = {'model': model, 'meta': meta}

        self.plant_transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.disease_transform = transforms.Compose([
            transforms.Resize((256, 256)), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def predict(self, image_path: str):
        img = Image.open(image_path).convert('RGB')
        x = self.plant_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.plant_model(x)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            top_idx = int(probs.argmax())
            top_conf = float(probs[top_idx])

        results = {
            'plant_prediction': top_idx,
            'plant_confidence': top_conf,
            'disease_prediction': None,
            'disease_confidence': None,
        }

        if top_conf < self.threshold:
            results['note'] = 'Uncertain/Review'
            return results

        # route to disease model
        species = top_idx
        # keys in disease models are species names; expect mapping outside
        # we assume user provided keys matching species index; try str(species)
        key = str(species)
        # fallback to first available model if exact key not found
        if key not in self.disease_models:
            if len(self.disease_models) == 0:
                results['note'] = 'No disease models loaded'
                return results
            # pick first
            key = list(self.disease_models.keys())[0]

        dm = self.disease_models[key]
        x2 = self.disease_transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out2 = dm['model'](x2)
            probs2 = F.softmax(out2, dim=1).cpu().numpy()[0]
            idx2 = int(probs2.argmax())
            conf2 = float(probs2[idx2])

        results['disease_prediction'] = idx2
        results['disease_confidence'] = conf2
        return results


if __name__ == '__main__':
    print('inference module. Instantiate InferencePipeline with checkpoints.')
