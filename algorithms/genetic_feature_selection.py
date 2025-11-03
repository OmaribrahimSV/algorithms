"""
الخوارزمية الجينية لاختيار الميزات
Genetic Algorithm for Feature Selection
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import random


class GeneticFeatureSelection:
    """خوارزمية جينية لاختيار الميزات"""
    
    def __init__(self, X, y, population_size=50, generations=30, mutation_rate=0.1, crossover_rate=0.8, callback=None):
        self.X = X
        self.y = y
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        self.callback = callback  # callback للبث في الوقت الفعلي
        
    def create_chromosome(self):
        """إنشاء كروموسوم عشوائي (ميزات مختارة)"""
        n_features = self.X.shape[1]
        # اختيار 30-70% من الميزات عشوائياً
        n_selected = random.randint(int(n_features * 0.3), int(n_features * 0.7))
        selected = random.sample(range(n_features), n_selected)
        chromosome = [0] * n_features
        for idx in selected:
            chromosome[idx] = 1
        return chromosome
    
    def fitness(self, chromosome):
        """حساب اللياقة (accuracy - penalty للعدد الكبير)"""
        selected_features = [i for i, bit in enumerate(chromosome) if bit == 1]
        
        if len(selected_features) == 0:
            return 0
        
        X_selected = self.X_scaled[:, selected_features]
        model = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)  # تقليل عدد الأشجار للسرعة
        
        try:
            scores = cross_val_score(model, X_selected, self.y, cv=5, scoring='accuracy', n_jobs=-1)
            accuracy = scores.mean()
            # معاقبة استخدام عدد كبير من الميزات
            penalty = len(selected_features) / self.X.shape[1] * 0.1
            return accuracy - penalty
        except:
            return 0
    
    def crossover(self, parent1, parent2):
        """تهجين (crossover) بين والدين"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    
    def mutate(self, chromosome):
        """طفر (mutation) عشوائي"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome
    
    def select_parents(self, population, fitness_scores):
        """اختيار الآباء بالانتخاب الدوراني"""
        max_fitness = max(fitness_scores)
        fitness_scores = [f + abs(min(fitness_scores)) + 1 for f in fitness_scores]
        total = sum(fitness_scores)
        probabilities = [f / total for f in fitness_scores]
        
        parent1 = np.random.choice(len(population), p=probabilities)
        parent2 = np.random.choice(len(population), p=probabilities)
        return population[parent1], population[parent2]
    
    def evolve(self):
        """تنفيذ الخوارزمية الجينية"""
        # إنشاء الجيل الأول
        population = [self.create_chromosome() for _ in range(self.population_size)]
        best_chromosome = None
        best_fitness = -1
        history = []
        
        for generation in range(self.generations):
            # حساب اللياقة
            fitness_scores = [self.fitness(chrom) for chrom in population]
            
            # تتبع الأفضل
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_fitness:
                best_fitness = fitness_scores[max_idx]
                best_chromosome = population[max_idx].copy()
            
            history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'n_features': sum(best_chromosome) if best_chromosome else 0
            })
            
            # بث التحديثات في الوقت الفعلي
            if self.callback:
                progress = (generation + 1) / self.generations * 100
                self.callback({
                    'type': 'progress',
                    'generation': generation + 1,
                    'total_generations': self.generations,
                    'progress': progress,
                    'best_fitness': best_fitness,
                    'avg_fitness': np.mean(fitness_scores),
                    'n_features': sum(best_chromosome) if best_chromosome else 0
                })
            
            # إنشاء الجيل الجديد
            new_population = [best_chromosome.copy()]  # الحفاظ على الأفضل
            
            while len(new_population) < self.population_size:
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        selected_features = [i for i, bit in enumerate(best_chromosome) if bit == 1]
        return selected_features, best_fitness, history

