import json
import os
import re
import math

# Original hand-picked list -- kept as a floor even when the mined weights
# file below is unavailable. Still real Windows APIs with obvious malicious
# intent (process injection/execution). "ConnectNetwork" was removed from
# the original list -- not a real Windows API name, had zero chance of ever
# matching real content.
_FALLBACK_SUSPICIOUS_APIS = [
    "NtCreateFile", "SetWindowsHookEx", "RegDeleteKey",
    "LdrLoadDll", "WinExec", "VirtualAllocEx",
    "WriteProcessMemory", "CreateRemoteThread", "CryptDecrypt",
]
_MINED_SIGNALS_PATH = "data/pacx_mining/mined_api_signals.json"
_MINED_STATIC_SIGNALS_PATH = "data/pacx_mining/mined_static_import_signals.json"


class PACXHeuristicAnalyzer:
    """
    Path 1: PAC-X (Prospect, Aspect, Context) Heuristic Analysis.
    Performs feature-based analysis independent of the GNN.
    """

    def __init__(self):
        # Evidence-weighted API scoring: each API's contribution to the
        # "aspect" score is its measured log-likelihood-ratio weight
        # (log2((p_malware+eps)/(p_benign+eps))), not a flat point-per-match.
        # A flat scheme forces an impossible choice between a short list
        # (misses most malware -- validated at 95.9% false-negative rate on
        # held-out real API-sequence data) or a longer one scored the same
        # way (drowns in false positives -- 65.4% on the same held-out data,
        # since common-but-somewhat-more-malware-associated calls like
        # NtAllocateVirtualMemory appear in 62.6% of REAL BENIGN samples
        # too). Weighting by real evidence strength fixed both: 80.47%
        # balanced accuracy on a held-out test split never used for mining
        # or threshold selection. See scripts/mine_pacx_api_signals.py and
        # data/pacx_mining/mined_api_signals.json (real Cuckoo Sandbox
        # traces, kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv).
        self.api_weights, self.aspect_threshold = self._load_mined_signals(_MINED_SIGNALS_PATH)
        # Separate weight table for STATIC import-table content (a raw PE
        # upload's real DIRECTORY_ENTRY_IMPORT names). Mined independently
        # from real static imports (18,551 real malware samples paired
        # against real Windows system-binary imports as the benign side --
        # see scripts/mine_pacx_static_import_signals.py) rather than reusing
        # the dynamic-trace weights with a hand-picked min_weight=2.5 floor
        # (that floor was a reasonable-but-unvalidated guess). Falls back to
        # the dynamic weights (still better than nothing) if the static
        # mining hasn't been run.
        self.static_api_weights, self.static_aspect_threshold = self._load_mined_signals(
            _MINED_STATIC_SIGNALS_PATH, fallback=(self.api_weights, self.aspect_threshold)
        )

    def _load_mined_signals(self, path, fallback=None):
        if os.path.exists(path):
            with open(path) as f:
                report = json.load(f)
            return report["weights"], report["decision_threshold"]
        if fallback is not None:
            return fallback
        # Fallback: flat weight 1.0 for the original hand-picked list, with
        # a threshold matching the old "3 matches -> aspect score 45" scale.
        return {api: 1.0 for api in _FALLBACK_SUSPICIOUS_APIS}, 3.0
        
    def _text_signal_ratio(self, content):
        """
        Fraction of alphabetic characters in the content. This heuristic only
        has signal on raw text/API logs; a numeric CSV feature row has almost
        no alphabetic content, so a low ratio means PAC-X has nothing real to
        evaluate here -- a "Benign" verdict in that case means "found
        nothing to match", not "verified clean", and must not be allowed to
        report high confidence.
        """
        if not content:
            return 0.0
        alpha_count = sum(1 for ch in content if ch.isalpha())
        return alpha_count / len(content)

    def _calculate_entropy(self, data):
        if not data:
            return 0
        entropy = 0
        for x in range(256):
            p_x = float(data.count(chr(x))) / len(data)
            if p_x > 0:
                entropy += - p_x * math.log(p_x, 2)
        return entropy

    def analyze(self, content, input_type=None):
        """Performs P-A-C analysis on raw content.

        `input_type` is a structural fact from the caller (json/csv/api/text),
        not a guess -- when it's "csv", the content is a PE-header feature
        row, not a behavioral log. This heuristic's keyword/entropy checks
        have no real signal there (column names like "f_SectionsMeanEntropy_0"
        aren't malicious indicators), so a low-signal cap always applies
        regardless of how much alphabetic text the header row happens to add.
        """
        score = 0
        findings = []
        category_scores = {
            "prospect": 0,
            "aspect": 0,
            "context": 0,
        }
        
        # 1. Prospect: Entropy & Header Analysis
        entropy = self._calculate_entropy(content)
        if entropy > 6.5:
            score += 30
            category_scores["prospect"] += 30
            findings.append("High entropy detected (Potential Packing/Encryption)")
            
        if "MZ" in content[:100] or "PE\0\0" in content:
            score += 10
            category_scores["prospect"] += 10
            findings.append("Executable header signatures identified")

        # 2. Aspect: Evidence-Weighted API Signature Detection. Every API
        # with a measured weight (see __init__) is checked, including
        # negative-weight ones (real evidence *against* malware, e.g. an API
        # that's actually more common in benign software) -- summing signed
        # log-likelihood-ratio evidence is the statistically correct way to
        # combine many weak indicators, unlike flat +N-per-match scoring.
        # A raw PE upload gives PAC-X a STATIC import table (APIs the binary
        # merely links against), a structurally different and weaker signal
        # than a DYNAMIC execution trace (an API actually observed being
        # called at runtime) -- so static content uses its own weight table,
        # mined from real static imports rather than reusing the dynamic
        # weights (see scripts/mine_pacx_static_import_signals.py).
        is_static = input_type == "pe_binary"
        api_weights = self.static_api_weights if is_static else self.api_weights
        aspect_threshold = self.static_aspect_threshold if is_static else self.aspect_threshold
        content_lower = content.lower()
        matched_apis = [api for api in api_weights if api.lower() in content_lower]
        if matched_apis:
            llr_sum = sum(api_weights[api] for api in matched_apis)
            # Scaled so reaching the real, validated decision threshold
            # (measured on held-out data -- see data/pacx_mining/
            # mined_api_signals.json / mined_static_import_signals.json)
            # contributes just over this function's overall malicious cutoff
            # (score > 40) on its own, and moves linearly with the margin
            # past that boundary. Margin-linear (not ratio-scaled) because
            # the static threshold is negative -- a ratio would flip sign.
            margin = llr_sum - aspect_threshold
            api_score = max(0.0, 41.0 + margin)
            score += api_score
            category_scores["aspect"] += api_score
            # Only surface the strong, positive-evidence matches in the
            # human-readable findings -- a full 307-API match list would be
            # noise; most matched APIs have small or negative weight and are
            # only meaningful summed together, not individually.
            strong_matches = sorted(
                (api for api in matched_apis if api_weights[api] > 1.0),
                key=lambda api: api_weights[api], reverse=True,
            )
            detected_apis = strong_matches
            if strong_matches:
                findings.append(f"Suspicious APIs detected: {', '.join(strong_matches)}")
        else:
            detected_apis = []

        # 3. Context: Behavioral String Patterns
        malicious_strings = ["http://", "powershell", "cmd.exe", "/c", "temp\\", "appdata"]
        content_lower = content.lower()
        detected_patterns = [p for p in malicious_strings if p in content_lower]
        if detected_patterns:
            context_score = len(detected_patterns) * 10
            score += context_score
            category_scores["context"] += context_score
            findings.append(f"Malicious context patterns: {', '.join(detected_patterns)}")

        # Final Classification
        malicious_score = min(score / 100.0, 1.0)
        prediction = "Malicious" if score > 40 else "Benign"
        # Confidence = malicious probability if malicious, benign probability if benign
        confidence = malicious_score if prediction == "Malicious" else (1.0 - malicious_score)

        # A "Benign" call with zero findings is an absence of evidence, not
        # evidence of absence -- cap the confidence instead of reporting 100%,
        # for either (a) content this heuristic structurally can't read (a
        # PE-header CSV row), or (b) text/api content that just has very
        # little in it to check in the first place.
        text_signal = self._text_signal_ratio(content)
        no_real_signal = (input_type == "csv") or text_signal < 0.05
        if prediction == "Benign" and no_real_signal:
            confidence = min(confidence, 0.5)

        total_category_score = sum(category_scores.values()) or 1
        breakdown = {
            key: round(value / total_category_score, 4)
            for key, value in category_scores.items()
        }
        
        return {
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "findings": findings,
            "entropy": round(entropy, 2),
            "detected_apis": detected_apis,
            "detected_patterns": detected_patterns,
            "breakdown": breakdown,
        }
