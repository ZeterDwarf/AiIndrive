import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class ScoringEngine:
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        self.cols = [
            'farm_area_ha', 'years_in_business', 'prev_subsidies',
            'efficiency_score', 'yield_growth', 'has_debts'
        ]
        self.trained = False

    def get_mock_data(self, n=1000):
        np.random.seed(42)
        df = pd.DataFrame({
            'farm_area_ha': np.random.uniform(50, 10000, n),
            'years_in_business': np.random.randint(1, 30, n),
            'prev_subsidies': np.random.randint(0, 15, n),
            'efficiency_score': np.random.normal(1.0, 0.5, n),
            'yield_growth': np.random.normal(5.0, 15.0, n),
            'has_debts': np.random.choice([0, 1], p=[0.8, 0.2], size=n)
        })
        
        base = df['yield_growth'] * 1.5 + df['years_in_business'] * 2 + df['efficiency_score'] * 20 - df['has_debts'] * 100 
        df['target'] = (base > np.median(base)).astype(int)
        return df

    def fit_model(self):
        data = self.get_mock_data()
        X = data[self.cols]
        y = data['target']
        
        X_scaled = self.scaler.fit_transform(X)
        self.clf.fit(X_scaled, y)
        self.trained = True

    def explain(self, app):
        res = []
        if app.get('has_debts', 0) == 1:
            res.append("Найдена налоговая задолженность - это критический риск.")
        else:
            res.append("Нет долгов по налогам (плюс к рейтингу).")
            
        eff = app.get('efficiency_score', 1.0)
        if eff > 1.2:
            res.append(f"Хорошая эффективность прошлых субсидий ({eff:.2f}).")
        elif eff < 0.8 and app.get('prev_subsidies', 0) > 0:
            res.append(f"Плохая эффективность прошлых субсидий ({eff:.2f}).")
            
        gw = app.get('yield_growth', 0)
        if gw > 5:
            res.append(f"Стабильно растет урожайность (+{gw}%).")
        elif gw < 0:
            res.append(f"Урожайность падает ({gw}%).")
            
        y = app.get('years_in_business', 0)
        if y >= 10:
             res.append("Большой опыт работы.")
             
        return res[:3]

    def predict_score(self, app):
        if not self.trained:
            self.fit_model()
            
        df = pd.DataFrame([app])
        for c in self.cols:
            if c not in df.columns:
                 df[c] = 0.0
                 
        X = df[self.cols]
        X_sc = self.scaler.transform(X)
        
        p = self.clf.predict_proba(X_sc)[0][1]
        sc = int(p * 100)
        
        dec = "Одобрено" if sc >= 60 else "Отказ"
        
        return {
            "score": sc,
            "decision": dec,
            "reasons": self.explain(app)
        }

if __name__ == "__main__":
    engine = ScoringEngine()
    
    app1 = {
        'farm_area_ha': 2500,
        'years_in_business': 15,
        'prev_subsidies': 3,
        'efficiency_score': 1.8, 
        'yield_growth': 12.5,
        'has_debts': 0
    }
    
    app2 = {
        'farm_area_ha': 500,
        'years_in_business': 3,
        'prev_subsidies': 1,
        'efficiency_score': 0.6, 
        'yield_growth': -2.0,
        'has_debts': 1 
    }
    
    print("--- Заявка 1 ---")
    print(app1)
    print(json.dumps(engine.predict_score(app1), indent=2, ensure_ascii=False))
    
    print("\n--- Заявка 2 ---")
    print(app2)
    print(json.dumps(engine.predict_score(app2), indent=2, ensure_ascii=False))
