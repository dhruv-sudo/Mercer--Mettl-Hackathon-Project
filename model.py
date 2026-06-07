def calculate_risk(face_score, voice_score):
    # 1. Calculate a trust/integrity score based on your features
    trust_score = (0.6 * face_score) + (0.4 * voice_score)
    
    # 2. Invert it: High trust means LOW risk (Risk = 1 - Trust)
    risk = 1.0 - trust_score
    
    # 3. Classify the risk accurately based on the remaining threat Level
    if risk > 0.7:
        label = "HIGH RISK"
    elif risk > 0.4:
        label = "MODERATE RISK"
    else:
        label = "LOW RISK"
        
    return {
        "risk_score": round(risk, 2),
        "label": label
    }
