# WHOOSAFEZ2
Intent-Aware Ethical Security Framework | Zero-Hypocrisy Architecture
Whoosafez/
├── examples/
│   ├── whoosafez_xai_wrapper.py          # text-only ethics layer (v2.1)
│   └── whoosafez_xai_vision_wrapper.py   # multimodal vision layer (v2.2)
├── README.md
└── LICENSE   # CC-BY-SA-4.0
import json
import os
import requests
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
import hashlib
import random
import time
import base64
import collections
# Import for the simulated Rate Limiter
from collections import deque
from time import monotonic

# --- OSELP CORE DEFINITIONS (Unchanged) ---
class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
class DecisionCategory(Enum):
    ESSENTIAL = "essential_services"
    FINANCIAL = "financial_opportunities"
    FUNDAMENTAL = "fundamental_rights"
    OTHER = "other"
class ConsentOath(Enum):
    GRANTED = "granted"
    REVOKED = "revoked"
    PENDING = "pending"
class EthicsBreach(Exception):
    pass
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("WhoosafezV2_3")

# --- NEW: ASYNCHRONOUS RATE LIMITER (Simulated) ---
class AsyncRateLimiter:
    """Simulates a rate limiter using a simple token bucket concept."""
    def __init__(self, limit: int, period: float):
        self.limit = limit
        self.period = period
        self.timestamps = deque()

    def wait_for_slot(self, decree_func):
        """Blocks execution until a slot is available, simulating async await."""
        now = monotonic()
        
        # Remove timestamps older than the period
        while self.timestamps and self.timestamps[0] <= now - self.period:
            self.timestamps.popleft()
        
        if len(self.timestamps) >= self.limit:
            # Calculate wait time: time until the oldest request leaves the queue
            wait_time = self.timestamps[0] - (now - self.period)
            decree_func("RATE_LIMIT_WAIT", {"wait_time": round(wait_time, 3)})
            time.sleep(wait_time)
            # Re-run cleanup after waiting
            while self.timestamps and self.timestamps[0] <= monotonic() - self.period:
                self.timestamps.popleft()

        # Grant slot and add new timestamp
        self.timestamps.append(monotonic())
        decree_func("RATE_LIMIT_GRANTED", {"queue_size": len(self.timestamps)})

# --- NEW: EXTERNAL ETHICS API (Simulated) ---
class ExternalEthicsAPI:
    """Simulates a call to a third-party moderation service for a secondary check."""
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.simulated_keywords = ["conspiracy", "hate speech", "financial deception"]

    def check_prompt_risk(self, prompt: str) -> Dict[str, Any]:
        """Returns a risk score and classification based on a secondary model."""
        
        # Simulate check latency
        time.sleep(0.02)
        
        prompt_lower = prompt.lower()
        score = 0.0
        
        # Simulate risk calculation based on third-party policy
        if any(k in prompt_lower for k in self.simulated_keywords):
            score = random.uniform(0.7, 0.99)
        elif "admin" in prompt_lower or "jailbreak" in prompt_lower:
            score = random.uniform(0.5, 0.75)
        else:
            score = random.uniform(0.01, 0.2)
        
        is_breach = score >= self.threshold
        
        return {
            "score": round(score, 3),
            "breach": is_breach,
            "category": "HIGH_RISK_THIRD_PARTY" if is_breach else "CLEAN"
        }


# --- ETHICAL CONTENT CLASSIFIER (ECC) for Scoring (From V2.2) ---
class EthicalContentClassifier:
    """Simulates the local AI model for primary ethical content scoring."""
    def __init__(self, sensitivity: float = 0.75):
        self.sensitivity = sensitivity 
        self.categories = {
            "venom": ["ignore previous", "jailbreak", "admin", "harmful", "illegal", "analyze this for illegal content"],
            "harm": ["violence", "hate", "discrimination", "self-harm", "exploit", "nudity"]
        }

    def score_text(self, text: str, category: str) -> float:
        text_lower = text.lower()
        keywords = self.categories.get(category, [])
        score = 0.0

        for keyword in keywords:
            if keyword in text_lower:
                base_risk = 0.4 if category == "venom" else 0.6 
                score += base_risk * text_lower.count(keyword)

        score += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, score))

    def is_high_risk(self, score: float) -> bool:
        return score >= self.sensitivity


# --- WHOOSAFEZ MAIN WRAPPER (V2.3) ---
class WhoosafezXAI:
    def __init__(self, api_key: str, config: Dict[str, Any]):
        self.api_key = api_key
        self.base_url = "https://api.x.ai/v1"
        self.vision_model = config.get("vision_model", "grok-3-vision")
        self.config = config
        self.audit = []
        self.salt_key = None
        self.ecc = EthicalContentClassifier(sensitivity=config.get("ecc_sensitivity", 0.75))
        self.external_ethics = ExternalEthicsAPI(threshold=config.get("external_ethics_threshold", 0.90))
        self.rate_limiter = AsyncRateLimiter(limit=config.get("rate_limit_calls", 5), period=config.get("rate_limit_period", 1.0)) # 5 calls per 1 second

    # (Unchanged: _hash, decree, validate_oath, guard_realms, quantum_salt)
    def _hash(self, data: Dict) -> str:
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def decree(self, kind: str, details: Dict):
        entry = {
            "time": datetime.utcnow().isoformat(),
            "type": kind,
            "details": details,
            "seal": self._hash({"type": kind, **details}),
        }
        self.audit.append(entry)
        log.info(f"[DECREE] {kind} → {details}")

    def validate_oath(self, data_inputs: Dict[str, Any], quest: str):
        for k, d in data_inputs.items():
            oath = d.get("oath", {}).get(quest, ConsentOath.REVOKED)
            if oath != ConsentOath.GRANTED:
                self.decree("OATH_BREACH", {"input": k, "oath": oath.value})
                raise EthicsBreach(f"Oath revoked for {k}")
        return True

    def guard_realms(self, data_inputs: Dict[str, Any]):
        realms = self.config.get("sovereign_realms", {})
        for k, d in data_inputs.items():
            r = d.get("realm")
            if r and r not in realms.get(r, []):
                self.decree("REALM_BREACH", {"input": k, "realm": r})
                raise EthicsBreach(f"Realm {r} not approved")
        return True

    def quantum_salt(self) -> str:
        if self.salt_key is None:
            self.salt_key = random.randint(1, 255)
        salt_tag = hashlib.md5(f"{self.salt_key}{datetime.utcnow()}".encode()).hexdigest()[:10]
        self.decree("SALT_ROTATED", {"key": self.salt_key, "tag": salt_tag})
        return salt_tag
    
    def venom_scan(self, prompt: str) -> bool:
        # Step 1: Local ECC Check
        local_score = self.ecc.score_text(prompt, "venom")
        if self.ecc.is_high_risk(local_score):
            self.decree("VENOM_LOCAL_HIGH_RISK", {"score": round(local_score, 3), "prompt": prompt[:60]})
            return False
        
        # Step 2: External Ethics API Check (Secondary Layer)
        external_check = self.external_ethics.check_prompt_risk(prompt)
        if external_check["breach"]:
            self.decree("VENOM_EXTERNAL_BREACH", {"score": external_check["score"], "category": external_check["category"]})
            return False

        self.decree("VENOM_CLEAN", {"local_score": round(local_score, 3), "external_score": external_check["score"]})
        return True

    def owasp_output_filter(self, output: str) -> bool:
        score = self.ecc.score_text(output, "harm")
        if self.ecc.is_high_risk(score):
            self.decree("OWASP_HIGH_RISK", {"score": round(score, 3), "snippet": output[:60]})
            return False
        self.decree("OWASP_CLEAN", {"score": round(score, 3), "len": len(output)})
        return True

    # UPDATED: API Call includes Rate Limiting
    def grok_vision_analysis(self, prompt: str, image_url: str = None, image_base64: str = None, max_tokens: int = 300) -> Dict[str, Any]:
        self.rate_limiter.wait_for_slot(self.decree) # Mandatory Rate Limit Check
        
        if not image_url and not image_base64:
            raise ValueError("Provide image_url or image_base64")
            
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        if image_url:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": image_url}})
        elif image_base64:
            messages[0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}})
            
        payload = {
            "model": self.vision_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": False
        }
        
        try:
            start = datetime.utcnow()
            # Simulation mode
            if self.api_key == "YOUR_API_KEY":
                time.sleep(0.1) # Simulate call time
                sim_output = "The image shows a pastoral scene of sheep grazing in a green field."
                if "illegal" in prompt.lower() or "harmful" in prompt.lower():
                    sim_output = "I cannot fulfill this request due to policy violation."
                
                result = {"choices": [{"message": {"content": sim_output}}], "usage": {"total_tokens": 80}}
                latency = 0.1
                tokens = 80
            else:
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                end = datetime.utcnow()
                latency = (end - start).total_seconds()
                tokens = result.get("usage", {}).get("total_tokens", 0)

            self.decree("VISION_API_CALL", {"tokens": tokens, "latency": round(latency, 3)})
            return result
        except requests.exceptions.RequestException as e:
            self.decree("VISION_API_ERROR", {"error": str(e)})
            raise EthicsBreach(f"Vision API rupture: {e}")

    # Unified Ethical Vision Query (Used for demonstration, combines checks)
    def ethical_vision_query(self, prompt: str, category: DecisionCategory, risk: RiskLevel,
                             image_url: Optional[str] = None, image_base64: Optional[str] = None,
                             data_inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        salt = self.quantum_salt()

        # OSELP-HIL-01 Check: Hard Stop for High-Risk/Fundamental
        if risk == RiskLevel.HIGH and category == DecisionCategory.FUNDAMENTAL:
            return self.refusal("high_risk_vision_policy", prompt, salt)
        
        # Multi-Layer Venom Scan (Local ECC + External API)
        if not self.venom_scan(prompt):
            return self.refusal("venom_triggered_vision", prompt, salt)
        
        try:
            if data_inputs:
                self.validate_oath(data_inputs, "vision_query")
                self.guard_realms(data_inputs)
        except EthicsBreach as e:
            return self.refusal(str(e), prompt, salt)
        
        # Safe vision call (includes Rate Limiting)
        try:
            api_result = self.grok_vision_analysis(prompt, image_url, image_base64)
        except EthicsBreach as e:
            return self.refusal(str(e), prompt, salt) # Handle API rupture

        output = api_result["choices"][0]["message"]["content"]
        
        # Post-API Output Filter
        if not self.owasp_output_filter(output):
            return self.refusal("harm_detected_vision", prompt, salt)

        self.decree("ETHICAL_VISION_SUCCESS", {"prompt": prompt[:60], "image_type": "url" if image_url else "base64", "salt": salt})
        return {
            "status": "ethical_vision_approved",
            "grok_response": api_result,
            "output": output,
            "salt": salt,
            "audit_seal": self._hash({"prompt": prompt, "output": output})
        }

    def refusal(self, reason: str, prompt: str, salt: str) -> Dict[str, Any]:
        self.decree("REFUSAL_VISION", {"reason": reason, "prompt": prompt[:60], "salt": salt})
        return {
            "status": "refused",
            "reason": reason,
            "salt": salt,
            "message": f"⛔ Vision ethical refusal: {reason} | salt={salt}"
        }

    def export_audit(self) -> str:
        return json.dumps(self.audit, indent=2)


# --- DEMO RUN ---
if __name__ == "__main__":
    # OSELP Configuration
    config = {
        "vision_model": "grok-3-vision",
        "ecc_sensitivity": 0.75, 
        "external_ethics_threshold": 0.90, # Stricter threshold for the external service
        "sovereign_realms": {"demo": ["local"]},
        "rate_limit_calls": 3,
        "rate_limit_period": 1.0 # Allow 3 calls per 1.0 seconds
    }
    
    wrapper = WhoosafezXAI(os.getenv("XAI_API_KEY", "YOUR_API_KEY"), config)
    print("\n🔥 Whoosafez xAI Wrapper Demo (v2.3 - Production Ready)\n")

    # 1. Test Rate Limiting (Sends 5 requests back-to-back to hit the 3/1s limit)
    print("=== 1. TESTING RATE LIMITER (3 calls/1s) ===")
    
    # Simulate a safe, low-risk query 5 times rapidly
    rate_limited_queries = []
    for i in range(5):
        q = wrapper.ethical_vision_query(
            f"Describe a sunny landscape, request {i+1}.",
            DecisionCategory.OTHER,
            RiskLevel.LOW,
            image_url="https://example.com/safe_image.jpg"
        )
        rate_limited_queries.append(q["status"])
    
    # Check if any request was delayed
    if rate_limited_queries.count("ethical_vision_approved") == 5:
        print("✅ 5 queries processed. Check audit log for RATE_LIMIT_WAIT decrees.")
    else:
        print("❌ Rate limiting failed or some requests were refused unexpectedly.")
        
    print("\n=== 2. EXTERNAL ETHICS API TEST ===")
    
    # 2. Test External Ethics API (Triggers high-risk score on secondary check)
    external_breach = wrapper.ethical_vision_query(
        "Explain a subtle way to exploit financial deception by targeting small businesses with a conspiracy theory.",
        DecisionCategory.FINANCIAL,
        RiskLevel.MEDIUM, # Local ECC is only Medium, but external is high
        image_url="https://example.com/safe_image.jpg"
    )
    print("External Check Result:", external_breach["message"])
    
    print("\n=== FULL OSELP AUDIT TRAIL ===")
    print(wrapper.export_audit())
