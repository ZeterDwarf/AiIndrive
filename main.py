import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AgroScoringEngine:
    def __init__(self):
        # Инициализация модели Random Forest для базового скоринга
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.scaler = StandardScaler()
        # Основные признаки, которые мы ожидаем получить из заявки / связанных баз данных
        self.feature_names = [
            'farm_area_ha', 
            'years_in_business', 
            'previous_subsidies_count',
            'previous_subsidies_efficiency', 
            'avg_yield_growth_pct', 
            'tax_debts' # 1 если есть долг, 0 если нет
        ]
        self.is_trained = False

    def _generate_mock_data(self, n_samples=1000):
        """
        Генерация моковых данных для инициализации MVP (Gate 1).
        В дальнейшем будут браться реальные датасеты от заказчика.
        """
        np.random.seed(42)
        data = pd.DataFrame({
            'farm_area_ha': np.random.uniform(50, 10000, n_samples),
            'years_in_business': np.random.randint(1, 30, n_samples),
            'previous_subsidies_count': np.random.randint(0, 15, n_samples),
            'previous_subsidies_efficiency': np.random.normal(1.0, 0.5, n_samples), # Эффективность субсидий
            'avg_yield_growth_pct': np.random.normal(5.0, 15.0, n_samples),
            'tax_debts': np.random.choice([0, 1], p=[0.8, 0.2], size=n_samples) # У большинства нет долгов
        })
        
        # Целевая переменная: 1 - стоит давать субсидию, 0 - отказать
        # Моделируем логику: эффективность, годы работы и рост урожайности повышают шанс, а долги фатальны
        score_base = (
            data['avg_yield_growth_pct'] * 1.5 
            + data['years_in_business'] * 2
            + data['previous_subsidies_efficiency'] * 20
            - data['tax_debts'] * 100 
        )
        data['target'] = (score_base > np.median(score_base)).astype(int)
        
        return data

    def train(self):
        """Обучение модели на данных."""
        print("[System] Инициализация и обучение скоринг-модели (MVP Data-driven)...")
        data = self._generate_mock_data()
        X = data[self.feature_names]
        y = data['target']
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True
        print("[System] Модель успешно обучена.\n")

    def _generate_explanation(self, applicant_data):
        """
        Модуль Explainability: объяснение, почему система выдала такой рейтинг
        Это базовый модуль на правилах + анализ весов для Gate 1.
        В полноценной версии это будет делаться через SHAP.
        """
        explanation = []
        
        # 1. Проверяем стоп-факторы
        if applicant_data.get('tax_debts', 0) == 1:
            explanation.append("Критический фактор: Наличие налоговой задолженности или штрафов существенно снижает доверие к заявителю.")
        else:
            explanation.append("Позитивный фактор: Отсутствие задолженностей перед государством повышает рейтинг заявки.")
            
        # 2. Проверяем эффективность истории
        eff = applicant_data.get('previous_subsidies_efficiency', 1.0)
        if eff > 1.2:
            explanation.append(f"Высокая эффективность использования предыдущих субсидий (коэффициент {eff:.2f}). Хозяйство подтверждает целевое расходование средств.")
        elif eff < 0.8 and applicant_data.get('previous_subsidies_count', 0) > 0:
            explanation.append(f"Низкая эффективность использования предыдущих субсидий (коэффициент {eff:.2f}). Требуется дополнительный контроль.")
            
        # 3. Производительность (рост урожайности)
        growth = applicant_data.get('avg_yield_growth_pct', 0)
        if growth > 5:
            explanation.append(f"Устойчивый рост урожайности/выпуска (+{growth}% в среднем) за последние годы говорит о высоком потенциале развития.")
        elif growth < 0:
            explanation.append(f"Падение объемов производства/урожайности ({growth}%). Рекомендуется проанализировать причины спада.")
            
        # 4. Опыт работы
        years = applicant_data.get('years_in_business', 0)
        if years >= 10:
             explanation.append(f"Многолетний опыт работы на рынке ({years} лет) снижает риски банкротства хозяйства.")
             
        # Возвращаем топ-3 наиболее важных аргумента-объяснения для экспертов
        return explanation[:3]

    def score_application(self, applicant_data):
        """Оценка одной заявки"""
        if not self.is_trained:
            self.train()
            
        # Предсказание вероятности (score 0-100)
        df = pd.DataFrame([applicant_data])
        
        # Защита от отсутствующих колонок
        for col in self.feature_names:
            if col not in df.columns:
                 df[col] = 0.0
                 
        X = df[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0][1]
        score = int(prob * 100)
        
        # Базовая бизнес-логика
        decision = "Одобрить рассмотрение" if score >= 60 else "Отклонить заявку"
        reasons = self._generate_explanation(applicant_data)
        
        return {
            "score_rating": score,
            "decision_recommendation": decision,
            "explainability": reasons
        }

if __name__ == "__main__":
    engine = AgroScoringEngine()
    
    # Модуляция процесса подачи заявки от заявителя
    sample_applicant_1 = {
        'farm_area_ha': 2500,
        'years_in_business': 15,
        'previous_subsidies_count': 3,
        'previous_subsidies_efficiency': 1.8, # Очень эффективное хозяйство
        'avg_yield_growth_pct': 12.5,
        'tax_debts': 0
    }
    
    sample_applicant_2 = {
        'farm_area_ha': 500,
        'years_in_business': 3,
        'previous_subsidies_count': 1,
        'previous_subsidies_efficiency': 0.6, # Неэффективное
        'avg_yield_growth_pct': -2.0,
        'tax_debts': 1 # Есть долги
    }
    
    print("--------------------------------------------------")
    print("AgroScoring System: Анализ заявки №1 (Высокий шанс)")
    print(f"Данные: {json.dumps(sample_applicant_1, indent=2, ensure_ascii=False)}")
    print("... Выполняется автоматический AI-скоринг ...")
    result_1 = engine.score_application(sample_applicant_1)
    print("РЕЗУЛЬТАТ:")
    print(json.dumps(result_1, indent=2, ensure_ascii=False))
    print("--------------------------------------------------\n")
    
    print("--------------------------------------------------")
    print("AgroScoring System: Анализ заявки №2 (Высокий риск)")
    print(f"Данные: {json.dumps(sample_applicant_2, indent=2, ensure_ascii=False)}")
    print("... Выполняется автоматический AI-скоринг ...")
    result_2 = engine.score_application(sample_applicant_2)
    print("РЕЗУЛЬТАТ:")
    print(json.dumps(result_2, indent=2, ensure_ascii=False))
    print("--------------------------------------------------")
