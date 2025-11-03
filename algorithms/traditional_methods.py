"""
الطرق التقليدية لاختيار الميزات
Traditional Feature Selection Methods
"""
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_classif, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


class TraditionalFeatureSelection:
    """الطرق التقليدية لاختيار الميزات"""
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def f_test(self, k=None):
        """اختيار الميزات باستخدام F-test"""
        if k is None:
            k = min(50, self.X.shape[1] // 2)
        selector = SelectKBest(f_classif, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.scores_
    
    def mutual_information(self, k=None):
        """اختيار الميزات باستخدام Mutual Information"""
        if k is None:
            k = min(50, self.X.shape[1] // 2)
        selector = SelectKBest(mutual_info_classif, k=k)
        X_selected = selector.fit_transform(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.scores_
    
    def rfe(self, n_features=None):
        """Recursive Feature Elimination"""
        if n_features is None:
            n_features = min(50, self.X.shape[1] // 2)
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(model, n_features_to_select=n_features)
        selector.fit(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, selector.ranking_
    
    def model_based(self, threshold='median'):
        """اختيار الميزات بناءً على نموذج"""
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(self.X, self.y)
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(self.X, self.y)
        selected_features = selector.get_support(indices=True).tolist()
        return selected_features, model.feature_importances_
    
    def evaluate(self, selected_features):
        """تقييم أداء الميزات المختارة"""
        if len(selected_features) == 0:
            return 0
        X_selected = self.X[:, selected_features]
        model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        scores = cross_val_score(model, X_selected, self.y, cv=5, scoring='accuracy', n_jobs=-1)
        return scores.mean()
    
    def compare_all(self, genetic_features, genetic_score):
        """مقارنة جميع الطرق"""
        results = {
            'genetic': {
                'features': genetic_features,
                'n_features': len(genetic_features),
                'accuracy': genetic_score
            }
        }
        
        # F-test
        f_features, f_scores = self.f_test(len(genetic_features))
        results['f_test'] = {
            'features': f_features,
            'n_features': len(f_features),
            'accuracy': self.evaluate(f_features),
            'scores': f_scores
        }
        
        # Mutual Information
        mi_features, mi_scores = self.mutual_information(len(genetic_features))
        results['mutual_info'] = {
            'features': mi_features,
            'n_features': len(mi_features),
            'accuracy': self.evaluate(mi_features),
            'scores': mi_scores
        }
        
        # RFE
        rfe_features, rfe_ranking = self.rfe(len(genetic_features))
        results['rfe'] = {
            'features': rfe_features,
            'n_features': len(rfe_features),
            'accuracy': self.evaluate(rfe_features)
        }
        
        # Model-based
        mb_features, mb_importances = self.model_based()
        results['model_based'] = {
            'features': mb_features,
            'n_features': len(mb_features),
            'accuracy': self.evaluate(mb_features),
            'importances': mb_importances
        }
        
        return results

