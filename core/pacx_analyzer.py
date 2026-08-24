import re
import math

class PACXHeuristicAnalyzer:
    """
    Path 1: PAC-X (Prospect, Aspect, Context) Heuristic Analysis.
    Performs feature-based analysis independent of the GNN.
    """
    
    def __init__(self):
        self.suspicious_apis = [
            "NTCreateFile", "SetWindowsHookEx", "RegDeleteKey", 
            "LdrLoadDll", "ConnectNetwork", "WinExec", "VirtualAllocEx",
            "WriteProcessMemory", "CreateRemoteThread", "CryptDecrypt"
        ]
        
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

        # 2. Aspect: API Signature Detection
        detected_apis = [api for api in self.suspicious_apis if api.lower() in content.lower()]
        if detected_apis:
            api_score = len(detected_apis) * 15
            score += api_score
            category_scores["aspect"] += api_score
            findings.append(f"Suspicious APIs detected: {', '.join(detected_apis)}")

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
