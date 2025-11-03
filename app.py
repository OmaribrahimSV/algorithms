"""
Flask Application for Genetic Feature Selection
تطبيق Flask لاختيار الميزات بالخوارزمية الجينية
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
from algorithms import GeneticFeatureSelection, TraditionalFeatureSelection

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')


@app.route('/get_columns', methods=['POST'])
def get_columns():
    """Get column names from uploaded CSV file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'يجب أن يكون الملف بصيغة CSV'}), 400
        
        # Read only columns
        df = pd.read_csv(file)
        columns = df.columns.tolist()
        
        return jsonify({
            'columns': columns, 
            'n_rows': int(df.shape[0]), 
            'n_cols': int(df.shape[1])
        })
    
    except Exception as e:
        return jsonify({'error': f'خطأ في قراءة الملف: {str(e)}'}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Process file upload and run genetic algorithm"""
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({'error': 'لم يتم رفع ملف'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'لم يتم اختيار ملف'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'يجب أن يكون الملف بصيغة CSV'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read data
        df = pd.read_csv(filepath)
        
        # Get target column from request
        target_column = request.form.get('target_column')
        if not target_column or target_column not in df.columns:
            target_column = df.columns[-1]  # Use last column as default
        
        # Prepare data
        X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Convert target to numeric if needed
        if y.dtype == 'object':
            y = pd.Categorical(y).codes
        
        # Get settings from request
        population_size = int(request.form.get('population_size', 50))
        generations = int(request.form.get('generations', 30))
        mutation_rate = float(request.form.get('mutation_rate', 0.1))
        
        # Run genetic algorithm (without callback for simplicity)
        ga = GeneticFeatureSelection(
            X.values, y.values,
            population_size=population_size,
            generations=generations,
            mutation_rate=mutation_rate,
            callback=None  # No real-time updates
        )
        genetic_features, genetic_score, history = ga.evolve()
        
        # Compare with traditional methods
        traditional = TraditionalFeatureSelection(X.values, y.values)
        comparison = traditional.compare_all(genetic_features, genetic_score)
        
        # Prepare results
        results = {
            'success': True,
            'data_info': {
                'n_rows': int(df.shape[0]),
                'n_features': int(X.shape[1]),
                'n_selected': len(genetic_features)
            },
            'genetic_algorithm': {
                'accuracy': float(genetic_score),
                'n_features': len(genetic_features),
                'selected_features': genetic_features,
                'feature_names': [X.columns[i] for i in genetic_features],
                'history': history
            },
            'comparison': {
                'methods': ['الخوارزمية الجينية', 'F-Test', 'Mutual Information', 'RFE', 'Model-Based'],
                'accuracies': [
                    float(comparison['genetic']['accuracy']),
                    float(comparison['f_test']['accuracy']),
                    float(comparison['mutual_info']['accuracy']),
                    float(comparison['rfe']['accuracy']),
                    float(comparison['model_based']['accuracy'])
                ],
                'n_features': [
                    comparison['genetic']['n_features'],
                    comparison['f_test']['n_features'],
                    comparison['mutual_info']['n_features'],
                    comparison['rfe']['n_features'],
                    comparison['model_based']['n_features']
                ]
            }
        }
        
        # Delete file after processing
        os.remove(filepath)
        
        return jsonify(results)
    
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'خطأ في المعالجة: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
