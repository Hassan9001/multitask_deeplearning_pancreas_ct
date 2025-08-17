# nnunetv2/training/nnUNetTrainer/nnUNetTrainer_MultiTask.py
from typing import Any
import os, json
from os.path import join, isfile
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_MultiTask(nnUNetTrainer):
    # IMPORTANT: match parent signature exactly (no extra params here!)
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device=device)

        # read your extras from env or set defaults
        self.cls_loss_weight = float(os.environ.get("MULTITASK_CLS_WEIGHT", 3.0))
        self.num_subtypes = int(os.environ.get("MULTITASK_NUM_SUBTYPES", 3))

        self._case_to_subtype = None
        self._val_case_logits, self._val_case_targets = defaultdict(list), {}
        self.bce_cls = nn.BCEWithLogitsLoss()
    def initialize(self):
        super().initialize()
        self.print_to_log_file(f"[MultiTask] cls_loss_weight={self.cls_loss_weight}, num_subtypes={self.num_subtypes}")

    def _load_case_to_subtype(self):
        if self._case_to_subtype is not None:
            return
        map_json = os.environ.get("MULTITASK_SUBTYPE_JSON", "")
        if not map_json:
            from nnunetv2.paths import nnUNet_raw
            map_json = join(nnUNet_raw, self.plans_manager.dataset_name, "subtype_mapping.json")
        if not isfile(map_json):
            raise FileNotFoundError(f"Set MULTITASK_SUBTYPE_JSON. Tried: {map_json}")
        with open(map_json, "r") as f:
            d = json.load(f)
        mapping = d["mapping"] if isinstance(d, dict) and "mapping" in d else d
        self._case_to_subtype = {k: int(v["subtype"]) for k, v in mapping.items() if v.get("subtype") is not None}

    def _cls_targets_from_keys(self, keys: list) -> torch.Tensor:
        self._load_case_to_subtype()
        t = []
        for k in keys:
            if k in self._case_to_subtype: t.append(self._case_to_subtype[k])
            elif "subtype0" in k: t.append(0)
            elif "subtype1" in k: t.append(1)
            elif "subtype2" in k: t.append(2)
            else: raise RuntimeError(f"Missing subtype for case {k}. Provide mapping JSON.")
        return torch.as_tensor(t, dtype=torch.long, device=self.device)

    def compute_loss(self, output: Any, data_dict: dict):
        # output is (seg, cls)
        seg_part = output[0] if isinstance(output, (list, tuple)) else output
        seg_logits = seg_part if isinstance(seg_part, (list, tuple)) else [seg_part]
        seg_loss = self.loss(seg_logits, data_dict["target"])

        if not (isinstance(output, (list, tuple)) and len(output) == 2):
            raise RuntimeError("Network must return (segmentation, classification_logits)")
        _, cls_logits = output

        keys = data_dict.get("keys") or data_dict.get("keys_str") or []
        cls_targets_idx = self._cls_targets_from_keys(keys)

        # BCE+one-hot (keep) — or swap to CE if you prefer
        cls_targets_oh = F.one_hot(cls_targets_idx, num_classes=self.num_subtypes).float()
        cls_loss = self.bce_cls(cls_logits, cls_targets_oh)
        # alt: cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_targets_idx)

        return seg_loss + self.cls_loss_weight * cls_loss

    def _on_validation_step_end(self):
        if not (isinstance(self.network_output, (list, tuple)) and len(self.network_output) == 2):
            return
        _, cls_logits = self.network_output
        probs = torch.softmax(cls_logits, dim=1).detach().cpu()
        keys = getattr(self, "_last_val_keys", None) or getattr(self, "_current_dataloader_keys", None)
        if not keys: return
        targets = self._cls_targets_from_keys(keys).detach().cpu().tolist()
        for i, k in enumerate(keys):
            self._val_case_logits[k].append(probs[i])
            self._val_case_targets[k] = int(targets[i])

    def _on_validation_epoch_end(self):
        if not self._val_case_logits:
            return
        import torch
        from collections import Counter
        correct = total = 0
        y_true, y_pred = [], []
        for k, chunks in self._val_case_logits.items():
            mean_prob = torch.stack(chunks).mean(0)
            pred = int(torch.argmax(mean_prob))
            y_pred.append(pred)
            y_true.append(self._val_case_targets[k])
            if pred == self._val_case_targets[k]:
                correct += 1
            total += 1
        # macro-F1
        num_classes = self.num_subtypes
        f1s = []
        for c in range(num_classes):
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall    = tp / (tp + fn) if (tp + fn) else 0.0
            f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            f1s.append(f1)
        macro_f1 = sum(f1s) / max(1, num_classes)

        acc = correct / max(1, total)
        self.print_to_log_file(f"[MultiTask] Val cls accuracy: {acc:.4f} over {total} cases | macro-F1: {macro_f1:.4f}")
        self._val_case_logits.clear(); self._val_case_targets.clear()

        
    def train_step(self, batch: dict):
        data = batch['data']
        target = batch['target']
        keys = self._extract_keys(batch)

        # --- move batch to same device as network ---
        dev = next(self.network.parameters()).device
        data = data.to(dev, non_blocking=True)
        if isinstance(target, (list, tuple)):
            target = [t.to(dev, non_blocking=True) for t in target]
        else:
            target = target.to(dev, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        output = self.network(data)          # (seg, cls)
        self.network_output = output

        # seg loss (deep supervision safe)
        seg_part = output[0] if isinstance(output, (list, tuple)) else output
        seg_logits = seg_part if isinstance(seg_part, (list, tuple)) else [seg_part]
        seg_loss = self.loss(seg_logits, target)

        # classification loss
        cls_loss = 0.0
        if isinstance(output, (list, tuple)) and len(output) == 2:
            _, cls_logits = output
            cls_targets_idx = self._cls_targets_from_keys(keys)
            cls_targets_oh = torch.nn.functional.one_hot(
                cls_targets_idx, num_classes=self.num_subtypes
            ).float()
            cls_loss = self.bce_cls(cls_logits, cls_targets_oh)

        loss = seg_loss + self.cls_loss_weight * cls_loss
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())


    @torch.no_grad()
    def validation_step(self, batch: dict):
        data = batch['data']
        target = batch['target']
        keys = self._extract_keys(batch)

        # --- move to device ---
        dev = next(self.network.parameters()).device
        data = data.to(dev, non_blocking=True)
        if isinstance(target, (list, tuple)):
            target = [t.to(dev, non_blocking=True) for t in target]
        else:
            target = target.to(dev, non_blocking=True)

        output = self.network(data)
        self.network_output = output

        seg_part = output[0] if isinstance(output, (list, tuple)) else output
        seg_logits = seg_part if isinstance(seg_part, (list, tuple)) else [seg_part]
        val_loss = self.loss(seg_logits, target)

        if isinstance(output, (list, tuple)) and len(output) == 2 and keys:
            _, cls_logits = output
            probs = torch.softmax(cls_logits, dim=1).detach().cpu()
            targets = self._cls_targets_from_keys(keys).detach().cpu().tolist()
            for i, k in enumerate(keys):
                self._val_case_logits[k].append(probs[i])
                self._val_case_targets[k] = int(targets[i])

        return float(val_loss.detach().cpu().item())

    
    def _extract_keys(self, batch) -> list[str]:
        # Pull keys from batch regardless of list / tuple / numpy array
        k = batch.get('keys', None)
        if k is None:
            k = batch.get('keys_str', None)
        if k is None:
            return []
        if isinstance(k, (list, tuple)):
            return [str(x) for x in k]
        # likely a numpy array or torch tensor
        try:
            return [str(x) for x in k.tolist()]
        except Exception:
            return [str(x) for x in k]
    def maybe_plot_network_architecture(self):
        was_training = self.network.training
        try:
            self.network.eval()
            return super().maybe_plot_network_architecture()
        finally:
            self.network.train(was_training)




# from typing import Any, Dict
# import os, csv, torch
# import torch.nn.functional as F
# from torch import nn
# from collections import defaultdict
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# class nnUNetTrainer_MultiTask(nnUNetTrainer):
#     def __init__(self, *args, cls_loss_weight: float = 3.0, num_subtypes: int = 3, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.cls_loss_weight = cls_loss_weight
#         self.num_subtypes = num_subtypes
#         self._case_to_subtype = {}
#         self._val_case_logits, self._val_case_targets = defaultdict(list), {}

#         self.bce_cls = nn.BCEWithLogitsLoss()

#     # def _load_case_to_subtype(self):
#     #     if self._case_to_subtype: return
#     #     import os, csv
#     #     csv_path = os.environ.get("MULTITASK_SUBTYPE_CSV", "")
#     #     if not csv_path:
#     #         ds_dir = os.path.dirname(self.plans_manager.dataset_json_file)
#     #         csv_path = os.path.join(ds_dir, f"{self.plans_manager.dataset_name}_subtypes.csv")
#     #     if os.path.isfile(csv_path):
#     #         with open(csv_path, newline="") as f:
#     #             reader = csv.DictReader(f)
#     #             for row in reader:
#     #                 self._case_to_subtype[row["case"]] = int(row["subtype"])
#     #     else:
#     #         print("[nnUNetTrainer_MultiTask] WARNING: subtype CSV not found; trying to infer from key names.")
#     def _load_case_to_subtype(self):
#         if self._case_to_subtype is not None:
#             return
#         # Allow override via env var; else default to the dataset’s nnUNet_raw folder structure you’re using
#         map_json = os.environ.get("MULTITASK_SUBTYPE_JSON", "")
#         if not map_json:
#             # Fallback guess: same dataset name, file named 'subtype_mapping.json' under nnUNet_raw/DATASET
#             # Users can always set MULTITASK_SUBTYPE_JSON for custom paths.
#             try:
#                 from nnunetv2.paths import nnUNet_raw
#                 map_json = join(nnUNet_raw, self.plans_manager.dataset_name, "subtype_mapping.json")
#             except Exception:
#                 map_json = ""  # last resort; will raise below if still missing
#         if not isfile(map_json):
#             raise FileNotFoundError(
#                 f"Could not find subtype mapping JSON. Set MULTITASK_SUBTYPE_JSON to its path. "
#                 f"Tried: {map_json}"
#             )
#         d = load_json(map_json)
#         # File schema: {"mapping": {"case_id": {"subtype": int, "split": "train/val/..."}, ...}}
#         mapping = d["mapping"]
#         self._case_to_subtype = {k: int(v["subtype"]) for k, v in mapping.items()}


#     def _cls_targets_from_keys(self, keys: list) -> torch.Tensor:
#         self._load_case_to_subtype()
#         targets = []
#         for k in keys:
#             if k in self._case_to_subtype: targets.append(self._case_to_subtype[k])
#             elif "subtype0" in k: targets.append(0)
#             elif "subtype1" in k: targets.append(1)
#             elif "subtype2" in k: targets.append(2)
#             else: raise RuntimeError(f"Missing subtype for case {k}. Provide a CSV mapping.")
#         return torch.as_tensor(targets, dtype=torch.long, device=self.device)

#     def compute_loss(self, output: Any, data_dict: dict):
#         seg_logits = output if isinstance(output, (list, tuple)) else [output]
#         seg_loss = self.loss(seg_logits, data_dict["target"])

#         if not (isinstance(self.network_output, (list, tuple)) and len(self.network_output) == 2):
#             raise RuntimeError("Network must return (segmentation, classification_logits)")
#         _, cls_logits = self.network_output

#         keys = data_dict.get("keys") or data_dict.get("keys_str")
#         cls_targets_idx = self._cls_targets_from_keys(keys)
#         cls_targets_oh = torch.nn.functional.one_hot(cls_targets_idx, num_classes=self.num_subtypes).float()
#         cls_loss = self.bce_cls(cls_logits, cls_targets_oh)
#         return seg_loss + self.cls_loss_weight * cls_loss

#     def _on_validation_step_end(self):
#         if not (isinstance(self.network_output, (list, tuple)) and len(self.network_output) == 2): return
#         _, cls_logits = self.network_output
#         probs = torch.softmax(cls_logits, dim=1)
#         keys = self._last_val_keys
#         targets = self._cls_targets_from_keys(keys)
#         for i, k in enumerate(keys):
#             self._val_case_logits[k].append(probs[i].detach().cpu())
#             self._val_case_targets[k] = int(targets[i])

#     def _on_validation_epoch_end(self):
#         if not self._val_case_logits: return
#         import torch
#         correct = 0
#         y_true, y_pred = [], []
#         for k, probs_list in self._val_case_logits.items():
#             mean_prob = torch.stack(probs_list).mean(dim=0)
#             pred = int(torch.argmax(mean_prob))
#             y_pred.append(pred)
#             y_true.append(self._val_case_targets[k])
#             if pred == self._val_case_targets[k]:
#                 correct += 1
#         acc = correct / max(1, len(y_true))
#         self.print_to_log_file(f"[MultiTask] Val classification accuracy: {acc:.4f} over {len(y_true)} cases")
#         self._val_case_logits.clear(); self._val_case_targets.clear()

# ##