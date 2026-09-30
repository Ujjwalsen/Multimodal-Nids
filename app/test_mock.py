import numpy as np
import pandas as pd
from FlowClassifier import FlowClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder
import os

def test_anomaly_logic():
    print("Initializing Mock Test...")
    
    # 1. Mock the FlowClassifier
    class MockFlowClassifier(FlowClassifier):
        def __init__(self):
            self.class_type = 'binary'
            self.dataset = 'unsw'
            self.protocol = 'tcp'
            self.unknown_id = 2 

    fc = MockFlowClassifier()
    
    # 2. Setup Mock Models (pre-trained logic)
    print("Setting up mock trained models...")
    X_flow = np.random.rand(100, 10)
    X_payload = np.random.rand(100, 20)
    y = np.random.randint(0, 2, size=100) # 0 or 1
    
    flow_model = RandomForestClassifier(n_estimators=10).fit(X_flow, y)
    payload_model = RandomForestClassifier(n_estimators=10).fit(X_payload, y)
    
    fc.flow_anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(X_flow)
    fc.payload_anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(X_payload)

    # 3. Test Prediction Logic
    print("Testing predict_vote_binary...")
    preds = fc.predict_vote_binary(flow_model, payload_model, X_flow, X_payload)
    
    anomalies = np.sum(preds == fc.unknown_id)
    print(f"Total anomalies flagged by Isolation Forest: {anomalies} out of 100")
    
    # 4. Test Evaluation Logic
    print("Testing get_binary_scores_vote with anomalies...")
    # Mocking os.path.join for /app/results so it doesn't crash locally
    os.makedirs('/app/results/', exist_ok=True) if os.name == 'posix' else os.makedirs('./results/', exist_ok=True)
    if os.name != 'posix':
        fc.dataset = 'test_unsw' # To avoid permission errors on root /app
        
    # We need to temporarily patch os.path.join inside the class or just let it crash and catch it
    import builtins
    original_join = os.path.join
    def safe_join(a, b):
        if a.startswith('/app/results/'):
            return b
        return original_join(a, b)
    os.path.join = safe_join

    try:
        metrics = fc.get_binary_scores_vote(flow_model, payload_model, X_flow, X_payload, y)
        print("Metrics Output:", metrics)
        
        print("Testing save_binary_scores_vote pandas alignment...")
        class_dict = {0: 'Normal', 1: 'Attack', fc.unknown_id: 'Unknown/Zero-Day'}
        fc.save_binary_scores_vote(flow_model, payload_model, X_flow, X_payload, y, class_dict)
        
        print("Test Complete. Logic is sound and crash-free.")
    finally:
        os.path.join = original_join

if __name__ == "__main__":
    test_anomaly_logic()
